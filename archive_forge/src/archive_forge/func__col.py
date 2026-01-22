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
def _col(self, table: pa.Table) -> pa.ChunkedArray:
    """
        Return the column referenced by the `InputRefExpr` operand.

        Parameters
        ----------
        table : pa.Table

        Returns
        -------
        pa.ChunkedArray
        """
    assert isinstance(self.operands[0], InputRefExpr)
    return self.operands[0].execute_arrow(table)