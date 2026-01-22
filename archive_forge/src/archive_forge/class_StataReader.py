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
class StataReader(StataParser, abc.Iterator):
    __doc__ = _stata_reader_doc
    _path_or_buf: IO[bytes]

    def __init__(self, path_or_buf: FilePath | ReadBuffer[bytes], convert_dates: bool=True, convert_categoricals: bool=True, index_col: str | None=None, convert_missing: bool=False, preserve_dtypes: bool=True, columns: Sequence[str] | None=None, order_categoricals: bool=True, chunksize: int | None=None, compression: CompressionOptions='infer', storage_options: StorageOptions | None=None) -> None:
        super().__init__()
        self._convert_dates = convert_dates
        self._convert_categoricals = convert_categoricals
        self._index_col = index_col
        self._convert_missing = convert_missing
        self._preserve_dtypes = preserve_dtypes
        self._columns = columns
        self._order_categoricals = order_categoricals
        self._original_path_or_buf = path_or_buf
        self._compression = compression
        self._storage_options = storage_options
        self._encoding = ''
        self._chunksize = chunksize
        self._using_iterator = False
        self._entered = False
        if self._chunksize is None:
            self._chunksize = 1
        elif not isinstance(chunksize, int) or chunksize <= 0:
            raise ValueError('chunksize must be a positive integer when set.')
        self._close_file: Callable[[], None] | None = None
        self._missing_values = False
        self._can_read_value_labels = False
        self._column_selector_set = False
        self._value_labels_read = False
        self._data_read = False
        self._dtype: np.dtype | None = None
        self._lines_read = 0
        self._native_byteorder = _set_endianness(sys.byteorder)

    def _ensure_open(self) -> None:
        """
        Ensure the file has been opened and its header data read.
        """
        if not hasattr(self, '_path_or_buf'):
            self._open_file()

    def _open_file(self) -> None:
        """
        Open the file (with compression options, etc.), and read header information.
        """
        if not self._entered:
            warnings.warn('StataReader is being used without using a context manager. Using StataReader as a context manager is the only supported method.', ResourceWarning, stacklevel=find_stack_level())
        handles = get_handle(self._original_path_or_buf, 'rb', storage_options=self._storage_options, is_text=False, compression=self._compression)
        if hasattr(handles.handle, 'seekable') and handles.handle.seekable():
            self._path_or_buf = handles.handle
            self._close_file = handles.close
        else:
            with handles:
                self._path_or_buf = BytesIO(handles.handle.read())
            self._close_file = self._path_or_buf.close
        self._read_header()
        self._setup_dtype()

    def __enter__(self) -> Self:
        """enter context manager"""
        self._entered = True
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None:
        if self._close_file:
            self._close_file()

    def close(self) -> None:
        """Close the handle if its open.

        .. deprecated: 2.0.0

           The close method is not part of the public API.
           The only supported way to use StataReader is to use it as a context manager.
        """
        warnings.warn('The StataReader.close() method is not part of the public API and will be removed in a future version without notice. Using StataReader as a context manager is the only supported method.', FutureWarning, stacklevel=find_stack_level())
        if self._close_file:
            self._close_file()

    def _set_encoding(self) -> None:
        """
        Set string encoding which depends on file version
        """
        if self._format_version < 118:
            self._encoding = 'latin-1'
        else:
            self._encoding = 'utf-8'

    def _read_int8(self) -> int:
        return struct.unpack('b', self._path_or_buf.read(1))[0]

    def _read_uint8(self) -> int:
        return struct.unpack('B', self._path_or_buf.read(1))[0]

    def _read_uint16(self) -> int:
        return struct.unpack(f'{self._byteorder}H', self._path_or_buf.read(2))[0]

    def _read_uint32(self) -> int:
        return struct.unpack(f'{self._byteorder}I', self._path_or_buf.read(4))[0]

    def _read_uint64(self) -> int:
        return struct.unpack(f'{self._byteorder}Q', self._path_or_buf.read(8))[0]

    def _read_int16(self) -> int:
        return struct.unpack(f'{self._byteorder}h', self._path_or_buf.read(2))[0]

    def _read_int32(self) -> int:
        return struct.unpack(f'{self._byteorder}i', self._path_or_buf.read(4))[0]

    def _read_int64(self) -> int:
        return struct.unpack(f'{self._byteorder}q', self._path_or_buf.read(8))[0]

    def _read_char8(self) -> bytes:
        return struct.unpack('c', self._path_or_buf.read(1))[0]

    def _read_int16_count(self, count: int) -> tuple[int, ...]:
        return struct.unpack(f'{self._byteorder}{'h' * count}', self._path_or_buf.read(2 * count))

    def _read_header(self) -> None:
        first_char = self._read_char8()
        if first_char == b'<':
            self._read_new_header()
        else:
            self._read_old_header(first_char)

    def _read_new_header(self) -> None:
        self._path_or_buf.read(27)
        self._format_version = int(self._path_or_buf.read(3))
        if self._format_version not in [117, 118, 119]:
            raise ValueError(_version_error.format(version=self._format_version))
        self._set_encoding()
        self._path_or_buf.read(21)
        self._byteorder = '>' if self._path_or_buf.read(3) == b'MSF' else '<'
        self._path_or_buf.read(15)
        self._nvar = self._read_uint16() if self._format_version <= 118 else self._read_uint32()
        self._path_or_buf.read(7)
        self._nobs = self._get_nobs()
        self._path_or_buf.read(11)
        self._data_label = self._get_data_label()
        self._path_or_buf.read(19)
        self._time_stamp = self._get_time_stamp()
        self._path_or_buf.read(26)
        self._path_or_buf.read(8)
        self._path_or_buf.read(8)
        self._seek_vartypes = self._read_int64() + 16
        self._seek_varnames = self._read_int64() + 10
        self._seek_sortlist = self._read_int64() + 10
        self._seek_formats = self._read_int64() + 9
        self._seek_value_label_names = self._read_int64() + 19
        self._seek_variable_labels = self._get_seek_variable_labels()
        self._path_or_buf.read(8)
        self._data_location = self._read_int64() + 6
        self._seek_strls = self._read_int64() + 7
        self._seek_value_labels = self._read_int64() + 14
        self._typlist, self._dtyplist = self._get_dtypes(self._seek_vartypes)
        self._path_or_buf.seek(self._seek_varnames)
        self._varlist = self._get_varlist()
        self._path_or_buf.seek(self._seek_sortlist)
        self._srtlist = self._read_int16_count(self._nvar + 1)[:-1]
        self._path_or_buf.seek(self._seek_formats)
        self._fmtlist = self._get_fmtlist()
        self._path_or_buf.seek(self._seek_value_label_names)
        self._lbllist = self._get_lbllist()
        self._path_or_buf.seek(self._seek_variable_labels)
        self._variable_labels = self._get_variable_labels()

    def _get_dtypes(self, seek_vartypes: int) -> tuple[list[int | str], list[str | np.dtype]]:
        self._path_or_buf.seek(seek_vartypes)
        typlist = []
        dtyplist = []
        for _ in range(self._nvar):
            typ = self._read_uint16()
            if typ <= 2045:
                typlist.append(typ)
                dtyplist.append(str(typ))
            else:
                try:
                    typlist.append(self.TYPE_MAP_XML[typ])
                    dtyplist.append(self.DTYPE_MAP_XML[typ])
                except KeyError as err:
                    raise ValueError(f'cannot convert stata types [{typ}]') from err
        return (typlist, dtyplist)

    def _get_varlist(self) -> list[str]:
        b = 33 if self._format_version < 118 else 129
        return [self._decode(self._path_or_buf.read(b)) for _ in range(self._nvar)]

    def _get_fmtlist(self) -> list[str]:
        if self._format_version >= 118:
            b = 57
        elif self._format_version > 113:
            b = 49
        elif self._format_version > 104:
            b = 12
        else:
            b = 7
        return [self._decode(self._path_or_buf.read(b)) for _ in range(self._nvar)]

    def _get_lbllist(self) -> list[str]:
        if self._format_version >= 118:
            b = 129
        elif self._format_version > 108:
            b = 33
        else:
            b = 9
        return [self._decode(self._path_or_buf.read(b)) for _ in range(self._nvar)]

    def _get_variable_labels(self) -> list[str]:
        if self._format_version >= 118:
            vlblist = [self._decode(self._path_or_buf.read(321)) for _ in range(self._nvar)]
        elif self._format_version > 105:
            vlblist = [self._decode(self._path_or_buf.read(81)) for _ in range(self._nvar)]
        else:
            vlblist = [self._decode(self._path_or_buf.read(32)) for _ in range(self._nvar)]
        return vlblist

    def _get_nobs(self) -> int:
        if self._format_version >= 118:
            return self._read_uint64()
        else:
            return self._read_uint32()

    def _get_data_label(self) -> str:
        if self._format_version >= 118:
            strlen = self._read_uint16()
            return self._decode(self._path_or_buf.read(strlen))
        elif self._format_version == 117:
            strlen = self._read_int8()
            return self._decode(self._path_or_buf.read(strlen))
        elif self._format_version > 105:
            return self._decode(self._path_or_buf.read(81))
        else:
            return self._decode(self._path_or_buf.read(32))

    def _get_time_stamp(self) -> str:
        if self._format_version >= 118:
            strlen = self._read_int8()
            return self._path_or_buf.read(strlen).decode('utf-8')
        elif self._format_version == 117:
            strlen = self._read_int8()
            return self._decode(self._path_or_buf.read(strlen))
        elif self._format_version > 104:
            return self._decode(self._path_or_buf.read(18))
        else:
            raise ValueError()

    def _get_seek_variable_labels(self) -> int:
        if self._format_version == 117:
            self._path_or_buf.read(8)
            return self._seek_value_label_names + 33 * self._nvar + 20 + 17
        elif self._format_version >= 118:
            return self._read_int64() + 17
        else:
            raise ValueError()

    def _read_old_header(self, first_char: bytes) -> None:
        self._format_version = int(first_char[0])
        if self._format_version not in [104, 105, 108, 111, 113, 114, 115]:
            raise ValueError(_version_error.format(version=self._format_version))
        self._set_encoding()
        self._byteorder = '>' if self._read_int8() == 1 else '<'
        self._filetype = self._read_int8()
        self._path_or_buf.read(1)
        self._nvar = self._read_uint16()
        self._nobs = self._get_nobs()
        self._data_label = self._get_data_label()
        self._time_stamp = self._get_time_stamp()
        if self._format_version > 108:
            typlist = [int(c) for c in self._path_or_buf.read(self._nvar)]
        else:
            buf = self._path_or_buf.read(self._nvar)
            typlistb = np.frombuffer(buf, dtype=np.uint8)
            typlist = []
            for tp in typlistb:
                if tp in self.OLD_TYPE_MAPPING:
                    typlist.append(self.OLD_TYPE_MAPPING[tp])
                else:
                    typlist.append(tp - 127)
        try:
            self._typlist = [self.TYPE_MAP[typ] for typ in typlist]
        except ValueError as err:
            invalid_types = ','.join([str(x) for x in typlist])
            raise ValueError(f'cannot convert stata types [{invalid_types}]') from err
        try:
            self._dtyplist = [self.DTYPE_MAP[typ] for typ in typlist]
        except ValueError as err:
            invalid_dtypes = ','.join([str(x) for x in typlist])
            raise ValueError(f'cannot convert stata dtypes [{invalid_dtypes}]') from err
        if self._format_version > 108:
            self._varlist = [self._decode(self._path_or_buf.read(33)) for _ in range(self._nvar)]
        else:
            self._varlist = [self._decode(self._path_or_buf.read(9)) for _ in range(self._nvar)]
        self._srtlist = self._read_int16_count(self._nvar + 1)[:-1]
        self._fmtlist = self._get_fmtlist()
        self._lbllist = self._get_lbllist()
        self._variable_labels = self._get_variable_labels()
        if self._format_version > 104:
            while True:
                data_type = self._read_int8()
                if self._format_version > 108:
                    data_len = self._read_int32()
                else:
                    data_len = self._read_int16()
                if data_type == 0:
                    break
                self._path_or_buf.read(data_len)
        self._data_location = self._path_or_buf.tell()

    def _setup_dtype(self) -> np.dtype:
        """Map between numpy and state dtypes"""
        if self._dtype is not None:
            return self._dtype
        dtypes = []
        for i, typ in enumerate(self._typlist):
            if typ in self.NUMPY_TYPE_MAP:
                typ = cast(str, typ)
                dtypes.append((f's{i}', f'{self._byteorder}{self.NUMPY_TYPE_MAP[typ]}'))
            else:
                dtypes.append((f's{i}', f'S{typ}'))
        self._dtype = np.dtype(dtypes)
        return self._dtype

    def _decode(self, s: bytes) -> str:
        s = s.partition(b'\x00')[0]
        try:
            return s.decode(self._encoding)
        except UnicodeDecodeError:
            encoding = self._encoding
            msg = f'\nOne or more strings in the dta file could not be decoded using {encoding}, and\nso the fallback encoding of latin-1 is being used.  This can happen when a file\nhas been incorrectly encoded by Stata or some other software. You should verify\nthe string values returned are correct.'
            warnings.warn(msg, UnicodeWarning, stacklevel=find_stack_level())
            return s.decode('latin-1')

    def _read_value_labels(self) -> None:
        self._ensure_open()
        if self._value_labels_read:
            return
        if self._format_version <= 108:
            self._value_labels_read = True
            self._value_label_dict: dict[str, dict[float, str]] = {}
            return
        if self._format_version >= 117:
            self._path_or_buf.seek(self._seek_value_labels)
        else:
            assert self._dtype is not None
            offset = self._nobs * self._dtype.itemsize
            self._path_or_buf.seek(self._data_location + offset)
        self._value_labels_read = True
        self._value_label_dict = {}
        while True:
            if self._format_version >= 117:
                if self._path_or_buf.read(5) == b'</val':
                    break
            slength = self._path_or_buf.read(4)
            if not slength:
                break
            if self._format_version <= 117:
                labname = self._decode(self._path_or_buf.read(33))
            else:
                labname = self._decode(self._path_or_buf.read(129))
            self._path_or_buf.read(3)
            n = self._read_uint32()
            txtlen = self._read_uint32()
            off = np.frombuffer(self._path_or_buf.read(4 * n), dtype=f'{self._byteorder}i4', count=n)
            val = np.frombuffer(self._path_or_buf.read(4 * n), dtype=f'{self._byteorder}i4', count=n)
            ii = np.argsort(off)
            off = off[ii]
            val = val[ii]
            txt = self._path_or_buf.read(txtlen)
            self._value_label_dict[labname] = {}
            for i in range(n):
                end = off[i + 1] if i < n - 1 else txtlen
                self._value_label_dict[labname][val[i]] = self._decode(txt[off[i]:end])
            if self._format_version >= 117:
                self._path_or_buf.read(6)
        self._value_labels_read = True

    def _read_strls(self) -> None:
        self._path_or_buf.seek(self._seek_strls)
        self.GSO = {'0': ''}
        while True:
            if self._path_or_buf.read(3) != b'GSO':
                break
            if self._format_version == 117:
                v_o = self._read_uint64()
            else:
                buf = self._path_or_buf.read(12)
                v_size = 2 if self._format_version == 118 else 3
                if self._byteorder == '<':
                    buf = buf[0:v_size] + buf[4:12 - v_size]
                else:
                    buf = buf[0:v_size] + buf[4 + v_size:]
                v_o = struct.unpack('Q', buf)[0]
            typ = self._read_uint8()
            length = self._read_uint32()
            va = self._path_or_buf.read(length)
            if typ == 130:
                decoded_va = va[0:-1].decode(self._encoding)
            else:
                decoded_va = str(va)
            self.GSO[str(v_o)] = decoded_va

    def __next__(self) -> DataFrame:
        self._using_iterator = True
        return self.read(nrows=self._chunksize)

    def get_chunk(self, size: int | None=None) -> DataFrame:
        """
        Reads lines from Stata file and returns as dataframe

        Parameters
        ----------
        size : int, defaults to None
            Number of lines to read.  If None, reads whole file.

        Returns
        -------
        DataFrame
        """
        if size is None:
            size = self._chunksize
        return self.read(nrows=size)

    @Appender(_read_method_doc)
    def read(self, nrows: int | None=None, convert_dates: bool | None=None, convert_categoricals: bool | None=None, index_col: str | None=None, convert_missing: bool | None=None, preserve_dtypes: bool | None=None, columns: Sequence[str] | None=None, order_categoricals: bool | None=None) -> DataFrame:
        self._ensure_open()
        if convert_dates is None:
            convert_dates = self._convert_dates
        if convert_categoricals is None:
            convert_categoricals = self._convert_categoricals
        if convert_missing is None:
            convert_missing = self._convert_missing
        if preserve_dtypes is None:
            preserve_dtypes = self._preserve_dtypes
        if columns is None:
            columns = self._columns
        if order_categoricals is None:
            order_categoricals = self._order_categoricals
        if index_col is None:
            index_col = self._index_col
        if nrows is None:
            nrows = self._nobs
        if self._nobs == 0 and nrows == 0:
            self._can_read_value_labels = True
            self._data_read = True
            data = DataFrame(columns=self._varlist)
            for i, col in enumerate(data.columns):
                dt = self._dtyplist[i]
                if isinstance(dt, np.dtype):
                    if dt.char != 'S':
                        data[col] = data[col].astype(dt)
            if columns is not None:
                data = self._do_select_columns(data, columns)
            return data
        if self._format_version >= 117 and (not self._value_labels_read):
            self._can_read_value_labels = True
            self._read_strls()
        assert self._dtype is not None
        dtype = self._dtype
        max_read_len = (self._nobs - self._lines_read) * dtype.itemsize
        read_len = nrows * dtype.itemsize
        read_len = min(read_len, max_read_len)
        if read_len <= 0:
            if convert_categoricals:
                self._read_value_labels()
            raise StopIteration
        offset = self._lines_read * dtype.itemsize
        self._path_or_buf.seek(self._data_location + offset)
        read_lines = min(nrows, self._nobs - self._lines_read)
        raw_data = np.frombuffer(self._path_or_buf.read(read_len), dtype=dtype, count=read_lines)
        self._lines_read += read_lines
        if self._lines_read == self._nobs:
            self._can_read_value_labels = True
            self._data_read = True
        if self._byteorder != self._native_byteorder:
            raw_data = raw_data.byteswap().view(raw_data.dtype.newbyteorder())
        if convert_categoricals:
            self._read_value_labels()
        if len(raw_data) == 0:
            data = DataFrame(columns=self._varlist)
        else:
            data = DataFrame.from_records(raw_data)
            data.columns = Index(self._varlist)
        if index_col is None:
            data.index = RangeIndex(self._lines_read - read_lines, self._lines_read)
        if columns is not None:
            data = self._do_select_columns(data, columns)
        for col, typ in zip(data, self._typlist):
            if isinstance(typ, int):
                data[col] = data[col].apply(self._decode)
        data = self._insert_strls(data)
        valid_dtypes = [i for i, dtyp in enumerate(self._dtyplist) if dtyp is not None]
        object_type = np.dtype(object)
        for idx in valid_dtypes:
            dtype = data.iloc[:, idx].dtype
            if dtype not in (object_type, self._dtyplist[idx]):
                data.isetitem(idx, data.iloc[:, idx].astype(dtype))
        data = self._do_convert_missing(data, convert_missing)
        if convert_dates:
            for i, fmt in enumerate(self._fmtlist):
                if any((fmt.startswith(date_fmt) for date_fmt in _date_formats)):
                    data.isetitem(i, _stata_elapsed_date_to_datetime_vec(data.iloc[:, i], fmt))
        if convert_categoricals and self._format_version > 108:
            data = self._do_convert_categoricals(data, self._value_label_dict, self._lbllist, order_categoricals)
        if not preserve_dtypes:
            retyped_data = []
            convert = False
            for col in data:
                dtype = data[col].dtype
                if dtype in (np.dtype(np.float16), np.dtype(np.float32)):
                    dtype = np.dtype(np.float64)
                    convert = True
                elif dtype in (np.dtype(np.int8), np.dtype(np.int16), np.dtype(np.int32)):
                    dtype = np.dtype(np.int64)
                    convert = True
                retyped_data.append((col, data[col].astype(dtype)))
            if convert:
                data = DataFrame.from_dict(dict(retyped_data))
        if index_col is not None:
            data = data.set_index(data.pop(index_col))
        return data

    def _do_convert_missing(self, data: DataFrame, convert_missing: bool) -> DataFrame:
        replacements = {}
        for i in range(len(data.columns)):
            fmt = self._typlist[i]
            if fmt not in self.VALID_RANGE:
                continue
            fmt = cast(str, fmt)
            nmin, nmax = self.VALID_RANGE[fmt]
            series = data.iloc[:, i]
            svals = series._values
            missing = (svals < nmin) | (svals > nmax)
            if not missing.any():
                continue
            if convert_missing:
                missing_loc = np.nonzero(np.asarray(missing))[0]
                umissing, umissing_loc = np.unique(series[missing], return_inverse=True)
                replacement = Series(series, dtype=object)
                for j, um in enumerate(umissing):
                    missing_value = StataMissingValue(um)
                    loc = missing_loc[umissing_loc == j]
                    replacement.iloc[loc] = missing_value
            else:
                dtype = series.dtype
                if dtype not in (np.float32, np.float64):
                    dtype = np.float64
                replacement = Series(series, dtype=dtype)
                if not replacement._values.flags['WRITEABLE']:
                    replacement = replacement.copy()
                replacement._values[missing] = np.nan
            replacements[i] = replacement
        if replacements:
            for idx, value in replacements.items():
                data.isetitem(idx, value)
        return data

    def _insert_strls(self, data: DataFrame) -> DataFrame:
        if not hasattr(self, 'GSO') or len(self.GSO) == 0:
            return data
        for i, typ in enumerate(self._typlist):
            if typ != 'Q':
                continue
            data.isetitem(i, [self.GSO[str(k)] for k in data.iloc[:, i]])
        return data

    def _do_select_columns(self, data: DataFrame, columns: Sequence[str]) -> DataFrame:
        if not self._column_selector_set:
            column_set = set(columns)
            if len(column_set) != len(columns):
                raise ValueError('columns contains duplicate entries')
            unmatched = column_set.difference(data.columns)
            if unmatched:
                joined = ', '.join(list(unmatched))
                raise ValueError(f'The following columns were not found in the Stata data set: {joined}')
            dtyplist = []
            typlist = []
            fmtlist = []
            lbllist = []
            for col in columns:
                i = data.columns.get_loc(col)
                dtyplist.append(self._dtyplist[i])
                typlist.append(self._typlist[i])
                fmtlist.append(self._fmtlist[i])
                lbllist.append(self._lbllist[i])
            self._dtyplist = dtyplist
            self._typlist = typlist
            self._fmtlist = fmtlist
            self._lbllist = lbllist
            self._column_selector_set = True
        return data[columns]

    def _do_convert_categoricals(self, data: DataFrame, value_label_dict: dict[str, dict[float, str]], lbllist: Sequence[str], order_categoricals: bool) -> DataFrame:
        """
        Converts categorical columns to Categorical type.
        """
        if not value_label_dict:
            return data
        cat_converted_data = []
        for col, label in zip(data, lbllist):
            if label in value_label_dict:
                vl = value_label_dict[label]
                keys = np.array(list(vl.keys()))
                column = data[col]
                key_matches = column.isin(keys)
                if self._using_iterator and key_matches.all():
                    initial_categories: np.ndarray | None = keys
                else:
                    if self._using_iterator:
                        warnings.warn(categorical_conversion_warning, CategoricalConversionWarning, stacklevel=find_stack_level())
                    initial_categories = None
                cat_data = Categorical(column, categories=initial_categories, ordered=order_categoricals)
                if initial_categories is None:
                    categories = []
                    for category in cat_data.categories:
                        if category in vl:
                            categories.append(vl[category])
                        else:
                            categories.append(category)
                else:
                    categories = list(vl.values())
                try:
                    cat_data = cat_data.rename_categories(categories)
                except ValueError as err:
                    vc = Series(categories, copy=False).value_counts()
                    repeated_cats = list(vc.index[vc > 1])
                    repeats = '-' * 80 + '\n' + '\n'.join(repeated_cats)
                    msg = f'\nValue labels for column {col} are not unique. These cannot be converted to\npandas categoricals.\n\nEither read the file with `convert_categoricals` set to False or use the\nlow level interface in `StataReader` to separately read the values and the\nvalue_labels.\n\nThe repeated labels are:\n{repeats}\n'
                    raise ValueError(msg) from err
                cat_series = Series(cat_data, index=data.index, copy=False)
                cat_converted_data.append((col, cat_series))
            else:
                cat_converted_data.append((col, data[col]))
        data = DataFrame(dict(cat_converted_data), copy=False)
        return data

    @property
    def data_label(self) -> str:
        """
        Return data label of Stata file.

        Examples
        --------
        >>> df = pd.DataFrame([(1,)], columns=["variable"])
        >>> time_stamp = pd.Timestamp(2000, 2, 29, 14, 21)
        >>> data_label = "This is a data file."
        >>> path = "/My_path/filename.dta"
        >>> df.to_stata(path, time_stamp=time_stamp,    # doctest: +SKIP
        ...             data_label=data_label,  # doctest: +SKIP
        ...             version=None)  # doctest: +SKIP
        >>> with pd.io.stata.StataReader(path) as reader:  # doctest: +SKIP
        ...     print(reader.data_label)  # doctest: +SKIP
        This is a data file.
        """
        self._ensure_open()
        return self._data_label

    @property
    def time_stamp(self) -> str:
        """
        Return time stamp of Stata file.
        """
        self._ensure_open()
        return self._time_stamp

    def variable_labels(self) -> dict[str, str]:
        """
        Return a dict associating each variable name with corresponding label.

        Returns
        -------
        dict

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=["col_1", "col_2"])
        >>> time_stamp = pd.Timestamp(2000, 2, 29, 14, 21)
        >>> path = "/My_path/filename.dta"
        >>> variable_labels = {"col_1": "This is an example"}
        >>> df.to_stata(path, time_stamp=time_stamp,  # doctest: +SKIP
        ...             variable_labels=variable_labels, version=None)  # doctest: +SKIP
        >>> with pd.io.stata.StataReader(path) as reader:  # doctest: +SKIP
        ...     print(reader.variable_labels())  # doctest: +SKIP
        {'index': '', 'col_1': 'This is an example', 'col_2': ''}
        >>> pd.read_stata(path)  # doctest: +SKIP
            index col_1 col_2
        0       0    1    2
        1       1    3    4
        """
        self._ensure_open()
        return dict(zip(self._varlist, self._variable_labels))

    def value_labels(self) -> dict[str, dict[float, str]]:
        """
        Return a nested dict associating each variable name to its value and label.

        Returns
        -------
        dict

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=["col_1", "col_2"])
        >>> time_stamp = pd.Timestamp(2000, 2, 29, 14, 21)
        >>> path = "/My_path/filename.dta"
        >>> value_labels = {"col_1": {3: "x"}}
        >>> df.to_stata(path, time_stamp=time_stamp,  # doctest: +SKIP
        ...             value_labels=value_labels, version=None)  # doctest: +SKIP
        >>> with pd.io.stata.StataReader(path) as reader:  # doctest: +SKIP
        ...     print(reader.value_labels())  # doctest: +SKIP
        {'col_1': {3: 'x'}}
        >>> pd.read_stata(path)  # doctest: +SKIP
            index col_1 col_2
        0       0    1    2
        1       1    x    4
        """
        if not self._value_labels_read:
            self._read_value_labels()
        return self._value_label_dict