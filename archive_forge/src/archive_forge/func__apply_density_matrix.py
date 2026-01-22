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
def _apply_density_matrix(self, state, device_wires):
    """Initialize the internal state in a specified mixed state.
        If not all the wires are specified in the full state :math:`\\rho`, remaining subsystem is filled by
        `\\mathrm{tr}_in(\\rho)`, which results in the full system state :math:`\\mathrm{tr}_{in}(\\rho) \\otimes \\rho_{in}`,
        where :math:`\\rho_{in}` is the argument `state` of this function and :math:`\\mathrm{tr}_{in}` is a partial
        trace over the subsystem to be replaced by this operation.

           Args:
               state (array[complex]): density matrix of length
                   ``(2**len(wires), 2**len(wires))``
               device_wires (Wires): wires that get initialized in the state
        """
    device_wires = self.map_wires(device_wires)
    state = qnp.asarray(state, dtype=self.C_DTYPE)
    state = qnp.reshape(state, (-1,))
    state_dim = 2 ** len(device_wires)
    dm_dim = state_dim ** 2
    if dm_dim != state.shape[0]:
        raise ValueError('Density matrix must be of length (2**wires, 2**wires)')
    if not qml.math.is_abstract(state) and (not qnp.allclose(qnp.trace(qnp.reshape(state, (state_dim, state_dim))), 1.0, atol=tolerance)):
        raise ValueError('Trace of density matrix is not equal one.')
    if len(device_wires) == self.num_wires and sorted(device_wires.labels) == list(device_wires.labels):
        self._state = qnp.reshape(state, [2] * 2 * self.num_wires)
        self._pre_rotated_state = self._state
    else:
        complement_wires = list(sorted(list(set(range(self.num_wires)) - set(device_wires))))
        sigma = self.density_matrix(Wires(complement_wires))
        rho = qnp.kron(sigma, state.reshape(state_dim, state_dim))
        rho = rho.reshape([2] * 2 * self.num_wires)
        left_axes = []
        right_axes = []
        complement_wires_count = len(complement_wires)
        for i in range(self.num_wires):
            if i in device_wires:
                index = device_wires.index(i)
                left_axes.append(complement_wires_count + index)
                right_axes.append(complement_wires_count + index + self.num_wires)
            elif i in complement_wires:
                index = complement_wires.index(i)
                left_axes.append(index)
                right_axes.append(index + self.num_wires)
        transpose_axes = left_axes + right_axes
        rho = qnp.transpose(rho, axes=transpose_axes)
        assert qml.math.is_abstract(rho) or qnp.allclose(qnp.trace(qnp.reshape(rho, (2 ** self.num_wires, 2 ** self.num_wires))), 1.0, atol=tolerance)
        self._state = qnp.asarray(rho, dtype=self.C_DTYPE)
        self._pre_rotated_state = self._state