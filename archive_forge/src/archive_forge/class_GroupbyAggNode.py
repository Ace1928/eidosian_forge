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
class GroupbyAggNode(DFAlgNode):
    """
    A node to represent a groupby aggregation operation.

    Parameters
    ----------
    base : DFAlgNode
        An aggregated frame.
    by : list of str
        A list of columns used for grouping.
    agg_exprs : dict
        Aggregates to compute.
    groupby_opts : dict
        Additional groupby parameters.

    Attributes
    ----------
    input : list of DFAlgNode
        Holds a single aggregated frame.
    by : list of str
        A list of columns used for grouping.
    agg_exprs : dict
        Aggregates to compute.
    groupby_opts : dict
        Additional groupby parameters.
    """

    def __init__(self, base, by, agg_exprs, groupby_opts):
        self.by = by
        self.agg_exprs = agg_exprs
        self.groupby_opts = groupby_opts
        self.input = [base]

    def copy(self):
        """
        Make a shallow copy of the node.

        Returns
        -------
        GroupbyAggNode
        """
        return GroupbyAggNode(self.input[0], self.by, self.agg_exprs, self.groupby_opts)

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
        return f'{prefix}AggNode:\n' + f'{prefix}  by: {self.by}\n' + f'{prefix}  aggs: {self.agg_exprs}\n' + f'{prefix}  groupby_opts: {self.groupby_opts}\n' + self._prints_input(prefix + '  ')