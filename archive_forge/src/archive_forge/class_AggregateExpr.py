import abc
from typing import Generator, Type, Union
import numpy as np
import pandas
import pyarrow as pa
import pyarrow.compute as pc
from pandas.core.dtypes.common import (
from modin.pandas.indexing import is_range_like
from modin.utils import _inherit_docstrings
from .dataframe.utils import ColNameCodec, to_arrow_type
class AggregateExpr(BaseExpr):
    """
    An aggregate operation expression.

    Parameters
    ----------
    agg : str
        Aggregate name.
    op : BaseExpr or list of BaseExpr
        Aggregate operand.
    distinct : bool, default: False
        Distinct modifier for 'count' aggregate.
    dtype : dtype, optional
        Aggregate data type. Computed if not specified.

    Attributes
    ----------
    agg : str
        Aggregate name.
    operands : list of BaseExpr
        Aggregate operands.
    distinct : bool
        Distinct modifier for 'count' aggregate.
    _dtype : dtype
        Aggregate data type.
    """

    def __init__(self, agg, op, distinct=False, dtype=None):
        if agg == 'nunique':
            self.agg = 'count'
            self.distinct = True
        else:
            self.agg = agg
            self.distinct = distinct
        self.operands = op if isinstance(op, list) else [op]
        self._dtype = dtype or _agg_dtype(self.agg, self.operands[0]._dtype)
        assert self._dtype is not None

    def copy(self):
        """
        Make a shallow copy of the expression.

        Returns
        -------
        AggregateExpr
        """
        return AggregateExpr(self.agg, self.operands, self.distinct, self._dtype)

    def __repr__(self):
        """
        Return a string representation of the expression.

        Returns
        -------
        str
        """
        if len(self.operands) == 1:
            return f'{self.agg}({self.operands[0]})[{self._dtype}]'
        return f'{self.agg}({self.operands})[{self._dtype}]'