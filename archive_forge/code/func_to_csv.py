from __future__ import annotations
import collections
from copy import deepcopy
import datetime as dt
from functools import partial
import gc
from json import loads
import operator
import pickle
import re
import sys
from typing import (
import warnings
import weakref
import numpy as np
from pandas._config import (
from pandas._libs import lib
from pandas._libs.lib import is_range_indexer
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
from pandas._typing import (
from pandas.compat import PYPY
from pandas.compat._constants import REF_COUNT
from pandas.compat._optional import import_optional_dependency
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import (
from pandas.core.dtypes.astype import astype_is_view
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import (
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core.array_algos.replace import should_use_regex
from pandas.core.arrays import ExtensionArray
from pandas.core.base import PandasObject
from pandas.core.construction import extract_array
from pandas.core.flags import Flags
from pandas.core.indexes.api import (
from pandas.core.internals import (
from pandas.core.internals.construction import (
from pandas.core.methods.describe import describe_ndframe
from pandas.core.missing import (
from pandas.core.reshape.concat import concat
from pandas.core.shared_docs import _shared_docs
from pandas.core.sorting import get_indexer_indexer
from pandas.core.window import (
from pandas.io.formats.format import (
from pandas.io.formats.printing import pprint_thing
@final
@deprecate_nonkeyword_arguments(version='3.0', allowed_args=['self', 'path_or_buf'], name='to_csv')
@doc(storage_options=_shared_docs['storage_options'], compression_options=_shared_docs['compression_options'] % 'path_or_buf')
def to_csv(self, path_or_buf: FilePath | WriteBuffer[bytes] | WriteBuffer[str] | None=None, sep: str=',', na_rep: str='', float_format: str | Callable | None=None, columns: Sequence[Hashable] | None=None, header: bool_t | list[str]=True, index: bool_t=True, index_label: IndexLabel | None=None, mode: str='w', encoding: str | None=None, compression: CompressionOptions='infer', quoting: int | None=None, quotechar: str='"', lineterminator: str | None=None, chunksize: int | None=None, date_format: str | None=None, doublequote: bool_t=True, escapechar: str | None=None, decimal: str='.', errors: OpenFileErrors='strict', storage_options: StorageOptions | None=None) -> str | None:
    """
        Write object to a comma-separated values (csv) file.

        Parameters
        ----------
        path_or_buf : str, path object, file-like object, or None, default None
            String, path object (implementing os.PathLike[str]), or file-like
            object implementing a write() function. If None, the result is
            returned as a string. If a non-binary file object is passed, it should
            be opened with `newline=''`, disabling universal newlines. If a binary
            file object is passed, `mode` might need to contain a `'b'`.
        sep : str, default ','
            String of length 1. Field delimiter for the output file.
        na_rep : str, default ''
            Missing data representation.
        float_format : str, Callable, default None
            Format string for floating point numbers. If a Callable is given, it takes
            precedence over other numeric formatting parameters, like decimal.
        columns : sequence, optional
            Columns to write.
        header : bool or list of str, default True
            Write out the column names. If a list of strings is given it is
            assumed to be aliases for the column names.
        index : bool, default True
            Write row names (index).
        index_label : str or sequence, or False, default None
            Column label for index column(s) if desired. If None is given, and
            `header` and `index` are True, then the index names are used. A
            sequence should be given if the object uses MultiIndex. If
            False do not print fields for index names. Use index_label=False
            for easier importing in R.
        mode : {{'w', 'x', 'a'}}, default 'w'
            Forwarded to either `open(mode=)` or `fsspec.open(mode=)` to control
            the file opening. Typical values include:

            - 'w', truncate the file first.
            - 'x', exclusive creation, failing if the file already exists.
            - 'a', append to the end of file if it exists.

        encoding : str, optional
            A string representing the encoding to use in the output file,
            defaults to 'utf-8'. `encoding` is not supported if `path_or_buf`
            is a non-binary file object.
        {compression_options}

               May be a dict with key 'method' as compression mode
               and other entries as additional compression options if
               compression mode is 'zip'.

               Passing compression options as keys in dict is
               supported for compression modes 'gzip', 'bz2', 'zstd', and 'zip'.
        quoting : optional constant from csv module
            Defaults to csv.QUOTE_MINIMAL. If you have set a `float_format`
            then floats are converted to strings and thus csv.QUOTE_NONNUMERIC
            will treat them as non-numeric.
        quotechar : str, default '\\"'
            String of length 1. Character used to quote fields.
        lineterminator : str, optional
            The newline character or character sequence to use in the output
            file. Defaults to `os.linesep`, which depends on the OS in which
            this method is called ('\\\\n' for linux, '\\\\r\\\\n' for Windows, i.e.).

            .. versionchanged:: 1.5.0

                Previously was line_terminator, changed for consistency with
                read_csv and the standard library 'csv' module.

        chunksize : int or None
            Rows to write at a time.
        date_format : str, default None
            Format string for datetime objects.
        doublequote : bool, default True
            Control quoting of `quotechar` inside a field.
        escapechar : str, default None
            String of length 1. Character used to escape `sep` and `quotechar`
            when appropriate.
        decimal : str, default '.'
            Character recognized as decimal separator. E.g. use ',' for
            European data.
        errors : str, default 'strict'
            Specifies how encoding and decoding errors are to be handled.
            See the errors argument for :func:`open` for a full list
            of options.

        {storage_options}

        Returns
        -------
        None or str
            If path_or_buf is None, returns the resulting csv format as a
            string. Otherwise returns None.

        See Also
        --------
        read_csv : Load a CSV file into a DataFrame.
        to_excel : Write DataFrame to an Excel file.

        Examples
        --------
        Create 'out.csv' containing 'df' without indices

        >>> df = pd.DataFrame({{'name': ['Raphael', 'Donatello'],
        ...                    'mask': ['red', 'purple'],
        ...                    'weapon': ['sai', 'bo staff']}})
        >>> df.to_csv('out.csv', index=False)  # doctest: +SKIP

        Create 'out.zip' containing 'out.csv'

        >>> df.to_csv(index=False)
        'name,mask,weapon\\nRaphael,red,sai\\nDonatello,purple,bo staff\\n'
        >>> compression_opts = dict(method='zip',
        ...                         archive_name='out.csv')  # doctest: +SKIP
        >>> df.to_csv('out.zip', index=False,
        ...           compression=compression_opts)  # doctest: +SKIP

        To write a csv file to a new folder or nested folder you will first
        need to create it using either Pathlib or os:

        >>> from pathlib import Path  # doctest: +SKIP
        >>> filepath = Path('folder/subfolder/out.csv')  # doctest: +SKIP
        >>> filepath.parent.mkdir(parents=True, exist_ok=True)  # doctest: +SKIP
        >>> df.to_csv(filepath)  # doctest: +SKIP

        >>> import os  # doctest: +SKIP
        >>> os.makedirs('folder/subfolder', exist_ok=True)  # doctest: +SKIP
        >>> df.to_csv('folder/subfolder/out.csv')  # doctest: +SKIP
        """
    df = self if isinstance(self, ABCDataFrame) else self.to_frame()
    formatter = DataFrameFormatter(frame=df, header=header, index=index, na_rep=na_rep, float_format=float_format, decimal=decimal)
    return DataFrameRenderer(formatter).to_csv(path_or_buf, lineterminator=lineterminator, sep=sep, encoding=encoding, errors=errors, compression=compression, quoting=quoting, columns=columns, index_label=index_label, mode=mode, chunksize=chunksize, quotechar=quotechar, date_format=date_format, doublequote=doublequote, escapechar=escapechar, storage_options=storage_options)