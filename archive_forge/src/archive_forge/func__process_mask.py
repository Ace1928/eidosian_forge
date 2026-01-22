from collections import abc
import pandas
from pandas.core.dtypes.common import _get_dtype, is_bool_dtype
from .calcite_algebra import (
from .dataframe.utils import ColNameCodec
from .df_algebra import (
from .expr import (
def _process_mask(self, op):
    """
        Translate ``MaskNode`` node.

        Parameters
        ----------
        op : MaskNode
            An operation to translate.
        """
    if op.row_labels is not None:
        raise NotImplementedError('row indices masking is not yet supported')
    frame = op.input[0]
    rowid_col = self._ref(frame, ColNameCodec.ROWID_COL_NAME)
    condition = build_row_idx_filter_expr(op.row_positions, rowid_col)
    self._push(CalciteFilterNode(condition))
    self._add_projection(frame)