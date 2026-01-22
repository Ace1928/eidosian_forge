from __future__ import annotations
from collections import abc
from datetime import (
from io import BytesIO
import os
import struct
import sys
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas._libs.lib import infer_dtype
from pandas._libs.writers import max_len_string_array
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas import (
from pandas.core.frame import DataFrame
from pandas.core.indexes.base import Index
from pandas.core.indexes.range import RangeIndex
from pandas.core.series import Series
from pandas.core.shared_docs import _shared_docs
from pandas.io.common import get_handle
@doc(storage_options=_shared_docs['storage_options'], compression_options=_shared_docs['compression_options'] % 'fname')
class StataWriter(StataParser):
    """
    A class for writing Stata binary dta files

    Parameters
    ----------
    fname : path (string), buffer or path object
        string, path object (pathlib.Path or py._path.local.LocalPath) or
        object implementing a binary write() functions. If using a buffer
        then the buffer will not be automatically closed after the file
        is written.
    data : DataFrame
        Input to save
    convert_dates : dict
        Dictionary mapping columns containing datetime types to stata internal
        format to use when writing the dates. Options are 'tc', 'td', 'tm',
        'tw', 'th', 'tq', 'ty'. Column can be either an integer or a name.
        Datetime columns that do not have a conversion type specified will be
        converted to 'tc'. Raises NotImplementedError if a datetime column has
        timezone information
    write_index : bool
        Write the index to Stata dataset.
    byteorder : str
        Can be ">", "<", "little", or "big". default is `sys.byteorder`
    time_stamp : datetime
        A datetime to use as file creation date.  Default is the current time
    data_label : str
        A label for the data set.  Must be 80 characters or smaller.
    variable_labels : dict
        Dictionary containing columns as keys and variable labels as values.
        Each label must be 80 characters or smaller.
    {compression_options}

        .. versionchanged:: 1.4.0 Zstandard support.

    {storage_options}

    value_labels : dict of dicts
        Dictionary containing columns as keys and dictionaries of column value
        to labels as values. The combined length of all labels for a single
        variable must be 32,000 characters or smaller.

        .. versionadded:: 1.4.0

    Returns
    -------
    writer : StataWriter instance
        The StataWriter instance has a write_file method, which will
        write the file to the given `fname`.

    Raises
    ------
    NotImplementedError
        * If datetimes contain timezone information
    ValueError
        * Columns listed in convert_dates are neither datetime64[ns]
          or datetime
        * Column dtype is not representable in Stata
        * Column listed in convert_dates is not in DataFrame
        * Categorical label contains more than 32,000 characters

    Examples
    --------
    >>> data = pd.DataFrame([[1.0, 1]], columns=['a', 'b'])
    >>> writer = StataWriter('./data_file.dta', data)
    >>> writer.write_file()

    Directly write a zip file
    >>> compression = {{"method": "zip", "archive_name": "data_file.dta"}}
    >>> writer = StataWriter('./data_file.zip', data, compression=compression)
    >>> writer.write_file()

    Save a DataFrame with dates
    >>> from datetime import datetime
    >>> data = pd.DataFrame([[datetime(2000,1,1)]], columns=['date'])
    >>> writer = StataWriter('./date_data_file.dta', data, {{'date' : 'tw'}})
    >>> writer.write_file()
    """
    _max_string_length = 244
    _encoding: Literal['latin-1', 'utf-8'] = 'latin-1'

    def __init__(self, fname: FilePath | WriteBuffer[bytes], data: DataFrame, convert_dates: dict[Hashable, str] | None=None, write_index: bool=True, byteorder: str | None=None, time_stamp: datetime | None=None, data_label: str | None=None, variable_labels: dict[Hashable, str] | None=None, compression: CompressionOptions='infer', storage_options: StorageOptions | None=None, *, value_labels: dict[Hashable, dict[float, str]] | None=None) -> None:
        super().__init__()
        self.data = data
        self._convert_dates = {} if convert_dates is None else convert_dates
        self._write_index = write_index
        self._time_stamp = time_stamp
        self._data_label = data_label
        self._variable_labels = variable_labels
        self._non_cat_value_labels = value_labels
        self._value_labels: list[StataValueLabel] = []
        self._has_value_labels = np.array([], dtype=bool)
        self._compression = compression
        self._output_file: IO[bytes] | None = None
        self._converted_names: dict[Hashable, str] = {}
        self._prepare_pandas(data)
        self.storage_options = storage_options
        if byteorder is None:
            byteorder = sys.byteorder
        self._byteorder = _set_endianness(byteorder)
        self._fname = fname
        self.type_converters = {253: np.int32, 252: np.int16, 251: np.int8}

    def _write(self, to_write: str) -> None:
        """
        Helper to call encode before writing to file for Python 3 compat.
        """
        self.handles.handle.write(to_write.encode(self._encoding))

    def _write_bytes(self, value: bytes) -> None:
        """
        Helper to assert file is open before writing.
        """
        self.handles.handle.write(value)

    def _prepare_non_cat_value_labels(self, data: DataFrame) -> list[StataNonCatValueLabel]:
        """
        Check for value labels provided for non-categorical columns. Value
        labels
        """
        non_cat_value_labels: list[StataNonCatValueLabel] = []
        if self._non_cat_value_labels is None:
            return non_cat_value_labels
        for labname, labels in self._non_cat_value_labels.items():
            if labname in self._converted_names:
                colname = self._converted_names[labname]
            elif labname in data.columns:
                colname = str(labname)
            else:
                raise KeyError(f"Can't create value labels for {labname}, it wasn't found in the dataset.")
            if not is_numeric_dtype(data[colname].dtype):
                raise ValueError(f"Can't create value labels for {labname}, value labels can only be applied to numeric columns.")
            svl = StataNonCatValueLabel(colname, labels, self._encoding)
            non_cat_value_labels.append(svl)
        return non_cat_value_labels

    def _prepare_categoricals(self, data: DataFrame) -> DataFrame:
        """
        Check for categorical columns, retain categorical information for
        Stata file and convert categorical data to int
        """
        is_cat = [isinstance(dtype, CategoricalDtype) for dtype in data.dtypes]
        if not any(is_cat):
            return data
        self._has_value_labels |= np.array(is_cat)
        get_base_missing_value = StataMissingValue.get_base_missing_value
        data_formatted = []
        for col, col_is_cat in zip(data, is_cat):
            if col_is_cat:
                svl = StataValueLabel(data[col], encoding=self._encoding)
                self._value_labels.append(svl)
                dtype = data[col].cat.codes.dtype
                if dtype == np.int64:
                    raise ValueError('It is not possible to export int64-based categorical data to Stata.')
                values = data[col].cat.codes._values.copy()
                if values.max() >= get_base_missing_value(dtype):
                    if dtype == np.int8:
                        dtype = np.dtype(np.int16)
                    elif dtype == np.int16:
                        dtype = np.dtype(np.int32)
                    else:
                        dtype = np.dtype(np.float64)
                    values = np.array(values, dtype=dtype)
                values[values == -1] = get_base_missing_value(dtype)
                data_formatted.append((col, values))
            else:
                data_formatted.append((col, data[col]))
        return DataFrame.from_dict(dict(data_formatted))

    def _replace_nans(self, data: DataFrame) -> DataFrame:
        """
        Checks floating point data columns for nans, and replaces these with
        the generic Stata for missing value (.)
        """
        for c in data:
            dtype = data[c].dtype
            if dtype in (np.float32, np.float64):
                if dtype == np.float32:
                    replacement = self.MISSING_VALUES['f']
                else:
                    replacement = self.MISSING_VALUES['d']
                data[c] = data[c].fillna(replacement)
        return data

    def _update_strl_names(self) -> None:
        """No-op, forward compatibility"""

    def _validate_variable_name(self, name: str) -> str:
        """
        Validate variable names for Stata export.

        Parameters
        ----------
        name : str
            Variable name

        Returns
        -------
        str
            The validated name with invalid characters replaced with
            underscores.

        Notes
        -----
        Stata 114 and 117 support ascii characters in a-z, A-Z, 0-9
        and _.
        """
        for c in name:
            if (c < 'A' or c > 'Z') and (c < 'a' or c > 'z') and (c < '0' or c > '9') and (c != '_'):
                name = name.replace(c, '_')
        return name

    def _check_column_names(self, data: DataFrame) -> DataFrame:
        """
        Checks column names to ensure that they are valid Stata column names.
        This includes checks for:
            * Non-string names
            * Stata keywords
            * Variables that start with numbers
            * Variables with names that are too long

        When an illegal variable name is detected, it is converted, and if
        dates are exported, the variable name is propagated to the date
        conversion dictionary
        """
        converted_names: dict[Hashable, str] = {}
        columns = list(data.columns)
        original_columns = columns[:]
        duplicate_var_id = 0
        for j, name in enumerate(columns):
            orig_name = name
            if not isinstance(name, str):
                name = str(name)
            name = self._validate_variable_name(name)
            if name in self.RESERVED_WORDS:
                name = '_' + name
            if '0' <= name[0] <= '9':
                name = '_' + name
            name = name[:min(len(name), 32)]
            if not name == orig_name:
                while columns.count(name) > 0:
                    name = '_' + str(duplicate_var_id) + name
                    name = name[:min(len(name), 32)]
                    duplicate_var_id += 1
                converted_names[orig_name] = name
            columns[j] = name
        data.columns = Index(columns)
        if self._convert_dates:
            for c, o in zip(columns, original_columns):
                if c != o:
                    self._convert_dates[c] = self._convert_dates[o]
                    del self._convert_dates[o]
        if converted_names:
            conversion_warning = []
            for orig_name, name in converted_names.items():
                msg = f'{orig_name}   ->   {name}'
                conversion_warning.append(msg)
            ws = invalid_name_doc.format('\n    '.join(conversion_warning))
            warnings.warn(ws, InvalidColumnName, stacklevel=find_stack_level())
        self._converted_names = converted_names
        self._update_strl_names()
        return data

    def _set_formats_and_types(self, dtypes: Series) -> None:
        self.fmtlist: list[str] = []
        self.typlist: list[int] = []
        for col, dtype in dtypes.items():
            self.fmtlist.append(_dtype_to_default_stata_fmt(dtype, self.data[col]))
            self.typlist.append(_dtype_to_stata_type(dtype, self.data[col]))

    def _prepare_pandas(self, data: DataFrame) -> None:
        data = data.copy()
        if self._write_index:
            temp = data.reset_index()
            if isinstance(temp, DataFrame):
                data = temp
        data = self._check_column_names(data)
        data = _cast_to_stata_types(data)
        data = self._replace_nans(data)
        self._has_value_labels = np.repeat(False, data.shape[1])
        non_cat_value_labels = self._prepare_non_cat_value_labels(data)
        non_cat_columns = [svl.labname for svl in non_cat_value_labels]
        has_non_cat_val_labels = data.columns.isin(non_cat_columns)
        self._has_value_labels |= has_non_cat_val_labels
        self._value_labels.extend(non_cat_value_labels)
        data = self._prepare_categoricals(data)
        self.nobs, self.nvar = data.shape
        self.data = data
        self.varlist = data.columns.tolist()
        dtypes = data.dtypes
        for col in data:
            if col in self._convert_dates:
                continue
            if lib.is_np_dtype(data[col].dtype, 'M'):
                self._convert_dates[col] = 'tc'
        self._convert_dates = _maybe_convert_to_int_keys(self._convert_dates, self.varlist)
        for key in self._convert_dates:
            new_type = _convert_datetime_to_stata_type(self._convert_dates[key])
            dtypes.iloc[key] = np.dtype(new_type)
        self._encode_strings()
        self._set_formats_and_types(dtypes)
        if self._convert_dates is not None:
            for key in self._convert_dates:
                if isinstance(key, int):
                    self.fmtlist[key] = self._convert_dates[key]

    def _encode_strings(self) -> None:
        """
        Encode strings in dta-specific encoding

        Do not encode columns marked for date conversion or for strL
        conversion. The strL converter independently handles conversion and
        also accepts empty string arrays.
        """
        convert_dates = self._convert_dates
        convert_strl = getattr(self, '_convert_strl', [])
        for i, col in enumerate(self.data):
            if i in convert_dates or col in convert_strl:
                continue
            column = self.data[col]
            dtype = column.dtype
            if dtype.type is np.object_:
                inferred_dtype = infer_dtype(column, skipna=True)
                if not (inferred_dtype == 'string' or len(column) == 0):
                    col = column.name
                    raise ValueError(f'Column `{col}` cannot be exported.\n\nOnly string-like object arrays\ncontaining all strings or a mix of strings and None can be exported.\nObject arrays containing only null values are prohibited. Other object\ntypes cannot be exported and must first be converted to one of the\nsupported types.')
                encoded = self.data[col].str.encode(self._encoding)
                if max_len_string_array(ensure_object(encoded._values)) <= self._max_string_length:
                    self.data[col] = encoded

    def write_file(self) -> None:
        """
        Export DataFrame object to Stata dta format.

        Examples
        --------
        >>> df = pd.DataFrame({"fully_labelled": [1, 2, 3, 3, 1],
        ...                    "partially_labelled": [1.0, 2.0, np.nan, 9.0, np.nan],
        ...                    "Y": [7, 7, 9, 8, 10],
        ...                    "Z": pd.Categorical(["j", "k", "l", "k", "j"]),
        ...                    })
        >>> path = "/My_path/filename.dta"
        >>> labels = {"fully_labelled": {1: "one", 2: "two", 3: "three"},
        ...           "partially_labelled": {1.0: "one", 2.0: "two"},
        ...           }
        >>> writer = pd.io.stata.StataWriter(path,
        ...                                  df,
        ...                                  value_labels=labels)  # doctest: +SKIP
        >>> writer.write_file()  # doctest: +SKIP
        >>> df = pd.read_stata(path)  # doctest: +SKIP
        >>> df  # doctest: +SKIP
            index fully_labelled  partially_labeled  Y  Z
        0       0            one                one  7  j
        1       1            two                two  7  k
        2       2          three                NaN  9  l
        3       3          three                9.0  8  k
        4       4            one                NaN 10  j
        """
        with get_handle(self._fname, 'wb', compression=self._compression, is_text=False, storage_options=self.storage_options) as self.handles:
            if self.handles.compression['method'] is not None:
                self._output_file, self.handles.handle = (self.handles.handle, BytesIO())
                self.handles.created_handles.append(self.handles.handle)
            try:
                self._write_header(data_label=self._data_label, time_stamp=self._time_stamp)
                self._write_map()
                self._write_variable_types()
                self._write_varnames()
                self._write_sortlist()
                self._write_formats()
                self._write_value_label_names()
                self._write_variable_labels()
                self._write_expansion_fields()
                self._write_characteristics()
                records = self._prepare_data()
                self._write_data(records)
                self._write_strls()
                self._write_value_labels()
                self._write_file_close_tag()
                self._write_map()
                self._close()
            except Exception as exc:
                self.handles.close()
                if isinstance(self._fname, (str, os.PathLike)) and os.path.isfile(self._fname):
                    try:
                        os.unlink(self._fname)
                    except OSError:
                        warnings.warn(f'This save was not successful but {self._fname} could not be deleted. This file is not valid.', ResourceWarning, stacklevel=find_stack_level())
                raise exc

    def _close(self) -> None:
        """
        Close the file if it was created by the writer.

        If a buffer or file-like object was passed in, for example a GzipFile,
        then leave this file open for the caller to close.
        """
        if self._output_file is not None:
            assert isinstance(self.handles.handle, BytesIO)
            bio, self.handles.handle = (self.handles.handle, self._output_file)
            self.handles.handle.write(bio.getvalue())

    def _write_map(self) -> None:
        """No-op, future compatibility"""

    def _write_file_close_tag(self) -> None:
        """No-op, future compatibility"""

    def _write_characteristics(self) -> None:
        """No-op, future compatibility"""

    def _write_strls(self) -> None:
        """No-op, future compatibility"""

    def _write_expansion_fields(self) -> None:
        """Write 5 zeros for expansion fields"""
        self._write(_pad_bytes('', 5))

    def _write_value_labels(self) -> None:
        for vl in self._value_labels:
            self._write_bytes(vl.generate_value_label(self._byteorder))

    def _write_header(self, data_label: str | None=None, time_stamp: datetime | None=None) -> None:
        byteorder = self._byteorder
        self._write_bytes(struct.pack('b', 114))
        self._write(byteorder == '>' and '\x01' or '\x02')
        self._write('\x01')
        self._write('\x00')
        self._write_bytes(struct.pack(byteorder + 'h', self.nvar)[:2])
        self._write_bytes(struct.pack(byteorder + 'i', self.nobs)[:4])
        if data_label is None:
            self._write_bytes(self._null_terminate_bytes(_pad_bytes('', 80)))
        else:
            self._write_bytes(self._null_terminate_bytes(_pad_bytes(data_label[:80], 80)))
        if time_stamp is None:
            time_stamp = datetime.now()
        elif not isinstance(time_stamp, datetime):
            raise ValueError('time_stamp should be datetime type')
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_lookup = {i + 1: month for i, month in enumerate(months)}
        ts = time_stamp.strftime('%d ') + month_lookup[time_stamp.month] + time_stamp.strftime(' %Y %H:%M')
        self._write_bytes(self._null_terminate_bytes(ts))

    def _write_variable_types(self) -> None:
        for typ in self.typlist:
            self._write_bytes(struct.pack('B', typ))

    def _write_varnames(self) -> None:
        for name in self.varlist:
            name = self._null_terminate_str(name)
            name = _pad_bytes(name[:32], 33)
            self._write(name)

    def _write_sortlist(self) -> None:
        srtlist = _pad_bytes('', 2 * (self.nvar + 1))
        self._write(srtlist)

    def _write_formats(self) -> None:
        for fmt in self.fmtlist:
            self._write(_pad_bytes(fmt, 49))

    def _write_value_label_names(self) -> None:
        for i in range(self.nvar):
            if self._has_value_labels[i]:
                name = self.varlist[i]
                name = self._null_terminate_str(name)
                name = _pad_bytes(name[:32], 33)
                self._write(name)
            else:
                self._write(_pad_bytes('', 33))

    def _write_variable_labels(self) -> None:
        blank = _pad_bytes('', 81)
        if self._variable_labels is None:
            for i in range(self.nvar):
                self._write(blank)
            return
        for col in self.data:
            if col in self._variable_labels:
                label = self._variable_labels[col]
                if len(label) > 80:
                    raise ValueError('Variable labels must be 80 characters or fewer')
                is_latin1 = all((ord(c) < 256 for c in label))
                if not is_latin1:
                    raise ValueError('Variable labels must contain only characters that can be encoded in Latin-1')
                self._write(_pad_bytes(label, 81))
            else:
                self._write(blank)

    def _convert_strls(self, data: DataFrame) -> DataFrame:
        """No-op, future compatibility"""
        return data

    def _prepare_data(self) -> np.rec.recarray:
        data = self.data
        typlist = self.typlist
        convert_dates = self._convert_dates
        if self._convert_dates is not None:
            for i, col in enumerate(data):
                if i in convert_dates:
                    data[col] = _datetime_to_stata_elapsed_vec(data[col], self.fmtlist[i])
        data = self._convert_strls(data)
        dtypes = {}
        native_byteorder = self._byteorder == _set_endianness(sys.byteorder)
        for i, col in enumerate(data):
            typ = typlist[i]
            if typ <= self._max_string_length:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', 'Downcasting object dtype arrays', category=FutureWarning)
                    dc = data[col].fillna('')
                data[col] = dc.apply(_pad_bytes, args=(typ,))
                stype = f'S{typ}'
                dtypes[col] = stype
                data[col] = data[col].astype(stype)
            else:
                dtype = data[col].dtype
                if not native_byteorder:
                    dtype = dtype.newbyteorder(self._byteorder)
                dtypes[col] = dtype
        return data.to_records(index=False, column_dtypes=dtypes)

    def _write_data(self, records: np.rec.recarray) -> None:
        self._write_bytes(records.tobytes())

    @staticmethod
    def _null_terminate_str(s: str) -> str:
        s += '\x00'
        return s

    def _null_terminate_bytes(self, s: str) -> bytes:
        return self._null_terminate_str(s).encode(self._encoding)