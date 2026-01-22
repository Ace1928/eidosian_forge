import functools
import itertools
from collections import defaultdict
from string import ascii_letters as ABC
import numpy as np
import pennylane as qml
import pennylane.math as qnp
from pennylane import (
from pennylane.measurements import CountsMP, MutualInfoMP, SampleMP, StateMP, VnEntropyMP, PurityMP
from pennylane.operation import Channel
from pennylane.ops.qubit.attributes import diagonal_in_z_basis
from pennylane.wires import Wires
from .._version import __version__
def _apply_channel_tensordot(self, kraus, wires):
    """Apply a quantum channel specified by a list of Kraus operators to subsystems of the
        quantum state. For a unitary gate, there is a single Kraus operator.

        Args:
            kraus (list[array]): Kraus operators
            wires (Wires): target wires
        """
    channel_wires = self.map_wires(wires)
    num_ch_wires = len(channel_wires)
    kraus_shape = [2] * (num_ch_wires * 2)
    kraus = [qnp.cast(qnp.reshape(k, kraus_shape), dtype=self.C_DTYPE) for k in kraus]
    row_wires_list = channel_wires.tolist()
    col_wires_list = [w + self.num_wires for w in row_wires_list]
    channel_col_ids = list(range(num_ch_wires, 2 * num_ch_wires))
    axes_left = [channel_col_ids, row_wires_list]
    axes_right = [col_wires_list, channel_col_ids]

    def _conjugate_state_with(k):
        """Perform the double tensor product k @ self._state @ k.conj().
            The `axes_left` and `axes_right` arguments are taken from the ambient variable space
            and `axes_right` is assumed to incorporate the tensor product and the transposition
            of k.conj() simultaneously."""
        return qnp.tensordot(qnp.tensordot(k, self._state, axes_left), qnp.conj(k), axes_right)
    if len(kraus) == 1:
        _state = _conjugate_state_with(kraus[0])
    else:
        _state = qnp.sum(qnp.stack([_conjugate_state_with(k) for k in kraus]), axis=0)
    source_left = list(range(num_ch_wires))
    dest_left = row_wires_list
    source_right = list(range(-num_ch_wires, 0))
    dest_right = col_wires_list
    self._state = qnp.moveaxis(_state, source_left + source_right, dest_left + dest_right)