from ... import symbol, ndarray
from ...base import string_types, numeric_types, _as_list
from ..block import Block, HybridBlock
from ..utils import _indent
from .. import tensor_types
from ..nn import LeakyReLU
class ZoneoutCell(ModifierCell):
    """Applies Zoneout on base cell."""

    def __init__(self, base_cell, zoneout_outputs=0.0, zoneout_states=0.0):
        assert not isinstance(base_cell, BidirectionalCell), "BidirectionalCell doesn't support zoneout since it doesn't support step. Please add ZoneoutCell to the cells underneath instead."
        assert not isinstance(base_cell, SequentialRNNCell) or not base_cell._bidirectional, "Bidirectional SequentialRNNCell doesn't support zoneout. Please add ZoneoutCell to the cells underneath instead."
        super(ZoneoutCell, self).__init__(base_cell)
        self.zoneout_outputs = zoneout_outputs
        self.zoneout_states = zoneout_states
        self._prev_output = None

    def __repr__(self):
        s = '{name}(p_out={zoneout_outputs}, p_state={zoneout_states}, {base_cell})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def _alias(self):
        return 'zoneout'

    def reset(self):
        super(ZoneoutCell, self).reset()
        self._prev_output = None

    def hybrid_forward(self, F, inputs, states):
        cell, p_outputs, p_states = (self.base_cell, self.zoneout_outputs, self.zoneout_states)
        next_output, next_states = cell(inputs, states)
        mask = lambda p, like: F.Dropout(F.ones_like(like), p=p)
        prev_output = self._prev_output
        if prev_output is None:
            prev_output = F.zeros_like(next_output)
        output = F.where(mask(p_outputs, next_output), next_output, prev_output) if p_outputs != 0.0 else next_output
        states = [F.where(mask(p_states, new_s), new_s, old_s) for new_s, old_s in zip(next_states, states)] if p_states != 0.0 else next_states
        self._prev_output = output
        return (output, states)