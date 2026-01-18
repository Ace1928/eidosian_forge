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
@deprecate_nonkeyword_arguments(version='3.0', allowed_args=['self', 'excel_writer'], name='to_excel')
@doc(klass='object', storage_options=_shared_docs['storage_options'], storage_options_versionadded='1.2.0')
def to_excel(self, excel_writer: FilePath | WriteExcelBuffer | ExcelWriter, sheet_name: str='Sheet1', na_rep: str='', float_format: str | None=None, columns: Sequence[Hashable] | None=None, header: Sequence[Hashable] | bool_t=True, index: bool_t=True, index_label: IndexLabel | None=None, startrow: int=0, startcol: int=0, engine: Literal['openpyxl', 'xlsxwriter'] | None=None, merge_cells: bool_t=True, inf_rep: str='inf', freeze_panes: tuple[int, int] | None=None, storage_options: StorageOptions | None=None, engine_kwargs: dict[str, Any] | None=None) -> None:
    """
        Write {klass} to an Excel sheet.

        To write a single {klass} to an Excel .xlsx file it is only necessary to
        specify a target file name. To write to multiple sheets it is necessary to
        create an `ExcelWriter` object with a target file name, and specify a sheet
        in the file to write to.

        Multiple sheets may be written to by specifying unique `sheet_name`.
        With all data written to the file it is necessary to save the changes.
        Note that creating an `ExcelWriter` object with a file name that already
        exists will result in the contents of the existing file being erased.

        Parameters
        ----------
        excel_writer : path-like, file-like, or ExcelWriter object
            File path or existing ExcelWriter.
        sheet_name : str, default 'Sheet1'
            Name of sheet which will contain DataFrame.
        na_rep : str, default ''
            Missing data representation.
        float_format : str, optional
            Format string for floating point numbers. For example
            ``float_format="%.2f"`` will format 0.1234 to 0.12.
        columns : sequence or list of str, optional
            Columns to write.
        header : bool or list of str, default True
            Write out the column names. If a list of string is given it is
            assumed to be aliases for the column names.
        index : bool, default True
            Write row names (index).
        index_label : str or sequence, optional
            Column label for index column(s) if desired. If not specified, and
            `header` and `index` are True, then the index names are used. A
            sequence should be given if the DataFrame uses MultiIndex.
        startrow : int, default 0
            Upper left cell row to dump data frame.
        startcol : int, default 0
            Upper left cell column to dump data frame.
        engine : str, optional
            Write engine to use, 'openpyxl' or 'xlsxwriter'. You can also set this
            via the options ``io.excel.xlsx.writer`` or
            ``io.excel.xlsm.writer``.

        merge_cells : bool, default True
            Write MultiIndex and Hierarchical Rows as merged cells.
        inf_rep : str, default 'inf'
            Representation for infinity (there is no native representation for
            infinity in Excel).
        freeze_panes : tuple of int (length 2), optional
            Specifies the one-based bottommost row and rightmost column that
            is to be frozen.
        {storage_options}

            .. versionadded:: {storage_options_versionadded}
        engine_kwargs : dict, optional
            Arbitrary keyword arguments passed to excel engine.

        See Also
        --------
        to_csv : Write DataFrame to a comma-separated values (csv) file.
        ExcelWriter : Class for writing DataFrame objects into excel sheets.
        read_excel : Read an Excel file into a pandas DataFrame.
        read_csv : Read a comma-separated values (csv) file into DataFrame.
        io.formats.style.Styler.to_excel : Add styles to Excel sheet.

        Notes
        -----
        For compatibility with :meth:`~DataFrame.to_csv`,
        to_excel serializes lists and dicts to strings before writing.

        Once a workbook has been saved it is not possible to write further
        data without rewriting the whole workbook.

        Examples
        --------

        Create, write to and save a workbook:

        >>> df1 = pd.DataFrame([['a', 'b'], ['c', 'd']],
        ...                    index=['row 1', 'row 2'],
        ...                    columns=['col 1', 'col 2'])
        >>> df1.to_excel("output.xlsx")  # doctest: +SKIP

        To specify the sheet name:

        >>> df1.to_excel("output.xlsx",
        ...              sheet_name='Sheet_name_1')  # doctest: +SKIP

        If you wish to write to more than one sheet in the workbook, it is
        necessary to specify an ExcelWriter object:

        >>> df2 = df1.copy()
        >>> with pd.ExcelWriter('output.xlsx') as writer:  # doctest: +SKIP
        ...     df1.to_excel(writer, sheet_name='Sheet_name_1')
        ...     df2.to_excel(writer, sheet_name='Sheet_name_2')

        ExcelWriter can also be used to append to an existing Excel file:

        >>> with pd.ExcelWriter('output.xlsx',
        ...                     mode='a') as writer:  # doctest: +SKIP
        ...     df1.to_excel(writer, sheet_name='Sheet_name_3')

        To set the library that is used to write the Excel file,
        you can pass the `engine` keyword (the default engine is
        automatically chosen depending on the file extension):

        >>> df1.to_excel('output1.xlsx', engine='xlsxwriter')  # doctest: +SKIP
        """
    if engine_kwargs is None:
        engine_kwargs = {}
    df = self if isinstance(self, ABCDataFrame) else self.to_frame()
    from pandas.io.formats.excel import ExcelFormatter
    formatter = ExcelFormatter(df, na_rep=na_rep, cols=columns, header=header, float_format=float_format, index=index, index_label=index_label, merge_cells=merge_cells, inf_rep=inf_rep)
    formatter.write(excel_writer, sheet_name=sheet_name, startrow=startrow, startcol=startcol, freeze_panes=freeze_panes, engine=engine, storage_options=storage_options, engine_kwargs=engine_kwargs)