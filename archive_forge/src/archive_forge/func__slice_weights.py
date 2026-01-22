import warnings
import functools
from .. import symbol, init, ndarray
from ..base import string_types, numeric_types
def _slice_weights(self, arr, li, lh):
    """slice fused rnn weights"""
    args = {}
    gate_names = self._gate_names
    directions = self._directions
    b = len(directions)
    p = 0
    for layer in range(self._num_layers):
        for direction in directions:
            for gate in gate_names:
                name = '%s%s%d_i2h%s_weight' % (self._prefix, direction, layer, gate)
                if layer > 0:
                    size = b * lh * lh
                    args[name] = arr[p:p + size].reshape((lh, b * lh))
                else:
                    size = li * lh
                    args[name] = arr[p:p + size].reshape((lh, li))
                p += size
            for gate in gate_names:
                name = '%s%s%d_h2h%s_weight' % (self._prefix, direction, layer, gate)
                size = lh ** 2
                args[name] = arr[p:p + size].reshape((lh, lh))
                p += size
    for layer in range(self._num_layers):
        for direction in directions:
            for gate in gate_names:
                name = '%s%s%d_i2h%s_bias' % (self._prefix, direction, layer, gate)
                args[name] = arr[p:p + lh]
                p += lh
            for gate in gate_names:
                name = '%s%s%d_h2h%s_bias' % (self._prefix, direction, layer, gate)
                args[name] = arr[p:p + lh]
                p += lh
    assert p == arr.size, 'Invalid parameters size for FusedRNNCell'
    return args