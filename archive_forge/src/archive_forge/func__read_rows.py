import codecs
import io
import os
import warnings
from csv import QUOTE_NONE
from typing import Callable, Optional, Sequence, Tuple, Union
import numpy as np
import pandas
import pandas._libs.lib as lib
from pandas.core.dtypes.common import is_list_like
from pandas.io.common import stringify_path
from modin.config import MinPartitionSize, NPartitions
from modin.core.io.file_dispatcher import FileDispatcher, OpenFile
from modin.core.io.text.utils import CustomNewlineIterator
from modin.core.storage_formats.pandas.utils import compute_chunksize
from modin.utils import _inherit_docstrings
@classmethod
def _read_rows(cls, f, nrows: int, quotechar: bytes=b'"', is_quoting: bool=True, outside_quotes: bool=True, encoding: str=None, newline: bytes=None):
    """
        Move the file offset at the specified amount of rows.

        Parameters
        ----------
        f : file-like object
            File handle that should be used for offset movement.
        nrows : int
            Number of rows to read.
        quotechar : bytes, default: b'"'
            Indicate quote in a file.
        is_quoting : bool, default: True
            Whether or not to consider quotes.
        outside_quotes : bool, default: True
            Whether the file pointer is within quotes or not at the time this function is called.
        encoding : str, optional
            Encoding of `f`.
        newline : bytes, optional
            Byte or sequence of bytes indicating line endings.

        Returns
        -------
        bool
            If file pointer reached the end of the file, but did not find closing quote
            returns `False`. `True` in any other case.
        int
            Number of rows that were read.
        """
    if nrows is not None and nrows <= 0:
        return (True, 0)
    rows_read = 0
    if encoding and ('utf' in encoding and '8' not in encoding or encoding == 'unicode_escape' or encoding.replace('-', '_') == 'utf_8_sig'):
        iterator = CustomNewlineIterator(f, newline)
    else:
        iterator = f
    for line in iterator:
        if is_quoting and line.count(quotechar) % 2:
            outside_quotes = not outside_quotes
        if outside_quotes:
            rows_read += 1
            if rows_read >= nrows:
                break
    if isinstance(iterator, CustomNewlineIterator):
        iterator.seek()
    if not outside_quotes:
        rows_read += 1
    return (outside_quotes, rows_read)