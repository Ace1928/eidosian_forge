from collections import abc
import pandas
from pandas.core.dtypes.common import _get_dtype, is_bool_dtype
from .calcite_algebra import (
from .dataframe.utils import ColNameCodec
from .df_algebra import (
from .expr import (
def _add_projection(self, frame):
    """
        Add a projection node to the resulting sequence.

        Added node simply selects all frame's columns. This method can be used
        to discard a virtual 'rowid' column provided by all scan nodes.

        Parameters
        ----------
        frame : HdkOnNativeDataframe
            An input frame for a projection.

        Returns
        -------
        CalciteProjectionNode
            Created projection node.
        """
    proj = CalciteProjectionNode(frame._table_cols, [self._ref(frame, col) for col in frame._table_cols])
    self._push(proj)
    return proj