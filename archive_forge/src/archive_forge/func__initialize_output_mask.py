from ...rnn import BidirectionalCell, SequentialRNNCell, ModifierCell, HybridRecurrentCell
from ...rnn.rnn_cell import _format_sequence, _get_begin_state, _mask_sequence_variable_length
from ... import tensor_types
from ....base import _as_list
def _initialize_output_mask(self, F, output):
    if self.drop_outputs and self.drop_outputs_mask is None:
        self.drop_outputs_mask = F.Dropout(F.ones_like(output), p=self.drop_outputs)