import abc
from typing import TYPE_CHECKING, Dict, List, Union
import numpy as np
import pandas
import pyarrow as pa
from pandas.core.dtypes.common import is_string_dtype
from modin.pandas.indexing import is_range_like
from modin.utils import _inherit_docstrings
from .dataframe.utils import EMPTY_ARROW_TABLE, ColNameCodec, get_common_arrow_type
from .db_worker import DbTable
from .expr import InputRefExpr, LiteralExpr, OpExpr
class SortNode(DFAlgNode):
    """
    A sort node to order frame's rows in a specified order.

    Parameters
    ----------
    frame : DFAlgNode
        Sorted frame.
    columns : list of str
        A list of key columns for a sort.
    ascending : list of bool
        Ascending or descending sort.
    na_position : {"first", "last"}
        "first" to put NULLs at the start of the result,
        "last" to put NULLs at the end of the result.

    Attributes
    ----------
    input : list of DFAlgNode
        Holds a single sorted frame.
    columns : list of str
        A list of key columns for a sort.
    ascending : list of bool
        Ascending or descending sort.
    na_position : {"first", "last"}
        "first" to put NULLs at the start of the result,
        "last" to put NULLs at the end of the result.
    """

    def __init__(self, frame, columns, ascending, na_position):
        self.input = [frame]
        self.columns = columns
        self.ascending = ascending
        self.na_position = na_position

    def copy(self):
        """
        Make a shallow copy of the node.

        Returns
        -------
        SortNode
        """
        return SortNode(self.input[0], self.columns, self.ascending, self.na_position)

    def _prints(self, prefix):
        """
        Return a string representation of the tree.

        Parameters
        ----------
        prefix : str
            A prefix to add at each string of the dump.

        Returns
        -------
        str
        """
        return f'{prefix}SortNode:\n' + f'{prefix}  Columns: {self.columns}\n' + f'{prefix}  Ascending: {self.ascending}\n' + f'{prefix}  NULLs position: {self.na_position}\n' + self._prints_input(prefix + '  ')