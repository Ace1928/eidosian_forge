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
def _pc(self, op: str, table: pa.Table) -> pa.ChunkedArray:
    """
        Perform the specified pyarrow.compute operation on the operands.

        Parameters
        ----------
        op : str
        table : pyarrow.Table

        Returns
        -------
        pyarrow.ChunkedArray
        """
    op = getattr(pc, op)
    val = self._op_value(0, table)
    for i in range(1, len(self.operands)):
        val = op(val, self._op_value(i, table))
    if not isinstance(val, pa.ChunkedArray):
        val = LiteralExpr(val).execute_arrow(table)
    if val.type != (at := to_arrow_type(self._dtype)):
        val = val.cast(at)
    return val