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
def _fold_literal(self, op, *args):
    """
        Fold literal expressions.

        Parameters
        ----------
        op : str

        *args : list

        Returns
        -------
        OpExpr or LiteralExpr
        """
    assert len(self.operands) == 1
    expr = self.operands[0]
    return getattr(expr, op)(*args) if isinstance(expr, LiteralExpr) else self