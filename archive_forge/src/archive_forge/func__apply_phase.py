import functools
import numpy as np
import pennylane as qml  # pylint: disable=unused-import
from pennylane import QutritDevice, QutritBasisState, DeviceError
from pennylane.wires import WireError
from pennylane.devices.default_qubit_legacy import _get_slice
from .._version import __version__
def _apply_phase(self, state, axes, index, phase, inverse=False):
    """Applies a phase onto the specified index along the axis specified in ``axes``.

        Args:
            state (array[complex]): input state
            axes (List[int]): target axes to apply transformation
            index (int): target index of axis to apply phase to
            phase (float): phase to apply
            inverse (bool): whether to apply the inverse phase

        Returns:
            array[complex]: output state
        """
    num_wires = len(state.shape)
    slices = [_get_slice(i, axes[0], num_wires) for i in range(3)]
    phase = self._conj(phase) if inverse else phase
    state_slices = [self._const_mul(phase if i == index else 1, state[slices[i]]) for i in range(3)]
    return self._stack(state_slices, axis=axes[0])