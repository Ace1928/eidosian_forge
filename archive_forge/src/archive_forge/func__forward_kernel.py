import re
from ... import ndarray, symbol
from .. import HybridBlock, tensor_types
from . import rnn_cell
from ...util import is_np_array
def _forward_kernel(self, F, inputs, states, sequence_length, **kwargs):
    """ forward using CUDNN or CPU kenrel"""
    swapaxes = F.np.swapaxes if is_np_array() else F.swapaxes
    if self._layout == 'NTC':
        inputs = swapaxes(inputs, 0, 1)
    if self._projection_size is None:
        params = (kwargs['{}{}_{}_{}'.format(d, l, g, t)].reshape(-1) for t in ['weight', 'bias'] for l in range(self._num_layers) for d in ['l', 'r'][:self._dir] for g in ['i2h', 'h2h'])
    else:
        params = (kwargs['{}{}_{}_{}'.format(d, l, g, t)].reshape(-1) for t in ['weight', 'bias'] for l in range(self._num_layers) for d in ['l', 'r'][:self._dir] for g in ['i2h', 'h2h', 'h2r'] if g != 'h2r' or t != 'bias')
    rnn_param_concat = F.np._internal.rnn_param_concat if is_np_array() else F._internal._rnn_param_concat
    params = rnn_param_concat(*params, dim=0)
    if self._use_sequence_length:
        rnn_args = states + [sequence_length]
    else:
        rnn_args = states
    rnn_fn = F.npx.rnn if is_np_array() else F.RNN
    rnn = rnn_fn(inputs, params, *rnn_args, use_sequence_length=self._use_sequence_length, state_size=self._hidden_size, projection_size=self._projection_size, num_layers=self._num_layers, bidirectional=self._dir == 2, p=self._dropout, state_outputs=True, mode=self._mode, lstm_state_clip_min=self._lstm_state_clip_min, lstm_state_clip_max=self._lstm_state_clip_max, lstm_state_clip_nan=self._lstm_state_clip_nan)
    if self._mode == 'lstm':
        outputs, states = (rnn[0], [rnn[1], rnn[2]])
    else:
        outputs, states = (rnn[0], [rnn[1]])
    if self._layout == 'NTC':
        outputs = swapaxes(outputs, 0, 1)
    return (outputs, states)