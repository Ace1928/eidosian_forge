from __future__ import annotations
from collections.abc import (
import datetime
from functools import partial
from io import BytesIO
import os
from textwrap import fill
from typing import (
import warnings
import zipfile
from pandas._config import config
from pandas._libs import lib
from pandas._libs.parsers import STR_NA_VALUES
from pandas.compat._optional import (
from pandas.errors import EmptyDataError
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend
from pandas.core.dtypes.common import (
from pandas.core.frame import DataFrame
from pandas.core.shared_docs import _shared_docs
from pandas.util.version import Version
from pandas.io.common import (
from pandas.io.excel._util import (
from pandas.io.parsers import TextParser
from pandas.io.parsers.readers import validate_integer
class BaseExcelReader(Generic[_WorkbookT]):
    book: _WorkbookT

    def __init__(self, filepath_or_buffer, storage_options: StorageOptions | None=None, engine_kwargs: dict | None=None) -> None:
        if engine_kwargs is None:
            engine_kwargs = {}
        if isinstance(filepath_or_buffer, bytes):
            filepath_or_buffer = BytesIO(filepath_or_buffer)
        self.handles = IOHandles(handle=filepath_or_buffer, compression={'method': None})
        if not isinstance(filepath_or_buffer, (ExcelFile, self._workbook_class)):
            self.handles = get_handle(filepath_or_buffer, 'rb', storage_options=storage_options, is_text=False)
        if isinstance(self.handles.handle, self._workbook_class):
            self.book = self.handles.handle
        elif hasattr(self.handles.handle, 'read'):
            self.handles.handle.seek(0)
            try:
                self.book = self.load_workbook(self.handles.handle, engine_kwargs)
            except Exception:
                self.close()
                raise
        else:
            raise ValueError('Must explicitly set engine if not passing in buffer or path for io.')

    @property
    def _workbook_class(self) -> type[_WorkbookT]:
        raise NotImplementedError

    def load_workbook(self, filepath_or_buffer, engine_kwargs) -> _WorkbookT:
        raise NotImplementedError

    def close(self) -> None:
        if hasattr(self, 'book'):
            if hasattr(self.book, 'close'):
                self.book.close()
            elif hasattr(self.book, 'release_resources'):
                self.book.release_resources()
        self.handles.close()

    @property
    def sheet_names(self) -> list[str]:
        raise NotImplementedError

    def get_sheet_by_name(self, name: str):
        raise NotImplementedError

    def get_sheet_by_index(self, index: int):
        raise NotImplementedError

    def get_sheet_data(self, sheet, rows: int | None=None):
        raise NotImplementedError

    def raise_if_bad_sheet_by_index(self, index: int) -> None:
        n_sheets = len(self.sheet_names)
        if index >= n_sheets:
            raise ValueError(f'Worksheet index {index} is invalid, {n_sheets} worksheets found')

    def raise_if_bad_sheet_by_name(self, name: str) -> None:
        if name not in self.sheet_names:
            raise ValueError(f"Worksheet named '{name}' not found")

    def _check_skiprows_func(self, skiprows: Callable, rows_to_use: int) -> int:
        """
        Determine how many file rows are required to obtain `nrows` data
        rows when `skiprows` is a function.

        Parameters
        ----------
        skiprows : function
            The function passed to read_excel by the user.
        rows_to_use : int
            The number of rows that will be needed for the header and
            the data.

        Returns
        -------
        int
        """
        i = 0
        rows_used_so_far = 0
        while rows_used_so_far < rows_to_use:
            if not skiprows(i):
                rows_used_so_far += 1
            i += 1
        return i

    def _calc_rows(self, header: int | Sequence[int] | None, index_col: int | Sequence[int] | None, skiprows: Sequence[int] | int | Callable[[int], object] | None, nrows: int | None) -> int | None:
        """
        If nrows specified, find the number of rows needed from the
        file, otherwise return None.


        Parameters
        ----------
        header : int, list of int, or None
            See read_excel docstring.
        index_col : int, str, list of int, or None
            See read_excel docstring.
        skiprows : list-like, int, callable, or None
            See read_excel docstring.
        nrows : int or None
            See read_excel docstring.

        Returns
        -------
        int or None
        """
        if nrows is None:
            return None
        if header is None:
            header_rows = 1
        elif is_integer(header):
            header = cast(int, header)
            header_rows = 1 + header
        else:
            header = cast(Sequence, header)
            header_rows = 1 + header[-1]
        if is_list_like(header) and index_col is not None:
            header = cast(Sequence, header)
            if len(header) > 1:
                header_rows += 1
        if skiprows is None:
            return header_rows + nrows
        if is_integer(skiprows):
            skiprows = cast(int, skiprows)
            return header_rows + nrows + skiprows
        if is_list_like(skiprows):

            def f(skiprows: Sequence, x: int) -> bool:
                return x in skiprows
            skiprows = cast(Sequence, skiprows)
            return self._check_skiprows_func(partial(f, skiprows), header_rows + nrows)
        if callable(skiprows):
            return self._check_skiprows_func(skiprows, header_rows + nrows)
        return None

    def parse(self, sheet_name: str | int | list[int] | list[str] | None=0, header: int | Sequence[int] | None=0, names: SequenceNotStr[Hashable] | range | None=None, index_col: int | Sequence[int] | None=None, usecols=None, dtype: DtypeArg | None=None, true_values: Iterable[Hashable] | None=None, false_values: Iterable[Hashable] | None=None, skiprows: Sequence[int] | int | Callable[[int], object] | None=None, nrows: int | None=None, na_values=None, verbose: bool=False, parse_dates: list | dict | bool=False, date_parser: Callable | lib.NoDefault=lib.no_default, date_format: dict[Hashable, str] | str | None=None, thousands: str | None=None, decimal: str='.', comment: str | None=None, skipfooter: int=0, dtype_backend: DtypeBackend | lib.NoDefault=lib.no_default, **kwds):
        validate_header_arg(header)
        validate_integer('nrows', nrows)
        ret_dict = False
        sheets: list[int] | list[str]
        if isinstance(sheet_name, list):
            sheets = sheet_name
            ret_dict = True
        elif sheet_name is None:
            sheets = self.sheet_names
            ret_dict = True
        elif isinstance(sheet_name, str):
            sheets = [sheet_name]
        else:
            sheets = [sheet_name]
        sheets = cast(Union[list[int], list[str]], list(dict.fromkeys(sheets).keys()))
        output = {}
        last_sheetname = None
        for asheetname in sheets:
            last_sheetname = asheetname
            if verbose:
                print(f'Reading sheet {asheetname}')
            if isinstance(asheetname, str):
                sheet = self.get_sheet_by_name(asheetname)
            else:
                sheet = self.get_sheet_by_index(asheetname)
            file_rows_needed = self._calc_rows(header, index_col, skiprows, nrows)
            data = self.get_sheet_data(sheet, file_rows_needed)
            if hasattr(sheet, 'close'):
                sheet.close()
            usecols = maybe_convert_usecols(usecols)
            if not data:
                output[asheetname] = DataFrame()
                continue
            is_list_header = False
            is_len_one_list_header = False
            if is_list_like(header):
                assert isinstance(header, Sequence)
                is_list_header = True
                if len(header) == 1:
                    is_len_one_list_header = True
            if is_len_one_list_header:
                header = cast(Sequence[int], header)[0]
            header_names = None
            if header is not None and is_list_like(header):
                assert isinstance(header, Sequence)
                header_names = []
                control_row = [True] * len(data[0])
                for row in header:
                    if is_integer(skiprows):
                        assert isinstance(skiprows, int)
                        row += skiprows
                    if row > len(data) - 1:
                        raise ValueError(f'header index {row} exceeds maximum index {len(data) - 1} of data.')
                    data[row], control_row = fill_mi_header(data[row], control_row)
                    if index_col is not None:
                        header_name, _ = pop_header_name(data[row], index_col)
                        header_names.append(header_name)
            has_index_names = False
            if is_list_header and (not is_len_one_list_header) and (index_col is not None):
                index_col_list: Sequence[int]
                if isinstance(index_col, int):
                    index_col_list = [index_col]
                else:
                    assert isinstance(index_col, Sequence)
                    index_col_list = index_col
                assert isinstance(header, Sequence)
                if len(header) < len(data):
                    potential_index_names = data[len(header)]
                    potential_data = [x for i, x in enumerate(potential_index_names) if not control_row[i] and i not in index_col_list]
                    has_index_names = all((x == '' or x is None for x in potential_data))
            if is_list_like(index_col):
                if header is None:
                    offset = 0
                elif isinstance(header, int):
                    offset = 1 + header
                else:
                    offset = 1 + max(header)
                if has_index_names:
                    offset += 1
                if offset < len(data):
                    assert isinstance(index_col, Sequence)
                    for col in index_col:
                        last = data[offset][col]
                        for row in range(offset + 1, len(data)):
                            if data[row][col] == '' or data[row][col] is None:
                                data[row][col] = last
                            else:
                                last = data[row][col]
            try:
                parser = TextParser(data, names=names, header=header, index_col=index_col, has_index_names=has_index_names, dtype=dtype, true_values=true_values, false_values=false_values, skiprows=skiprows, nrows=nrows, na_values=na_values, skip_blank_lines=False, parse_dates=parse_dates, date_parser=date_parser, date_format=date_format, thousands=thousands, decimal=decimal, comment=comment, skipfooter=skipfooter, usecols=usecols, dtype_backend=dtype_backend, **kwds)
                output[asheetname] = parser.read(nrows=nrows)
                if header_names:
                    output[asheetname].columns = output[asheetname].columns.set_names(header_names)
            except EmptyDataError:
                output[asheetname] = DataFrame()
            except Exception as err:
                err.args = (f'{err.args[0]} (sheet: {asheetname})', *err.args[1:])
                raise err
        if last_sheetname is None:
            raise ValueError('Sheet name is an empty list')
        if ret_dict:
            return output
        else:
            return output[last_sheetname]