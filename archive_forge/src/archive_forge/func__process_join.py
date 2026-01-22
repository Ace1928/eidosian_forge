from collections import abc
import pandas
from pandas.core.dtypes.common import _get_dtype, is_bool_dtype
from .calcite_algebra import (
from .dataframe.utils import ColNameCodec
from .df_algebra import (
from .expr import (
def _process_join(self, op):
    """
        Translate ``JoinNode`` node.

        Parameters
        ----------
        op : JoinNode
            An operation to translate.
        """
    self.has_join = True
    node = CalciteJoinNode(left_id=self._input_node(0).id, right_id=self._input_node(1).id, how=op.how, condition=self._translate(op.condition))
    self._push(node)
    self._push(CalciteProjectionNode(op.exprs.keys(), [self._translate(val) for val in op.exprs.values()]))