from collections import abc
import pandas
from pandas.core.dtypes.common import _get_dtype, is_bool_dtype
from .calcite_algebra import (
from .dataframe.utils import ColNameCodec
from .df_algebra import (
from .expr import (
def _process_sort(self, op):
    """
        Translate ``SortNode`` node.

        Parameters
        ----------
        op : SortNode
            An operation to translate.
        """
    frame = op.input[0]
    if not isinstance(self._input_node(0), CalciteProjectionNode):
        proj = self._add_projection(frame)
        self._input_ctx().replace_input_node(frame, proj, frame._table_cols)
    nulls = op.na_position.upper()
    collations = []
    for col, asc in zip(op.columns, op.ascending):
        ascending = 'ASCENDING' if asc else 'DESCENDING'
        collations.append(CalciteCollation(self._ref_idx(frame, col), ascending, nulls))
    self._push(CalciteSortNode(collations))