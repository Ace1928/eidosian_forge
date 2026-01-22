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
def _fold_invert(self) -> Union['OpExpr', LiteralExpr]:
    """
        Fold invert expression.

        Returns
        -------
        OpExpr or LiteralExpr
        """
    assert len(self.operands) == 1
    op = self.operands[0]
    if isinstance(op, LiteralExpr):
        return LiteralExpr(~op.val, op._dtype)
    if isinstance(op, OpExpr):
        if op.op == 'IS NULL':
            return OpExpr('IS NOT NULL', op.operands, op._dtype)
        if op.op == 'IS NOT NULL':
            return OpExpr('IS NULL', op.operands, op._dtype)
    return self