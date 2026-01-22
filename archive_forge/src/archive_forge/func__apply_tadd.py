import functools
import numpy as np
import pennylane as qml  # pylint: disable=unused-import
from pennylane import QutritDevice, QutritBasisState, DeviceError
from pennylane.wires import WireError
from pennylane.devices.default_qubit_legacy import _get_slice
from .._version import __version__
def _apply_tadd(self, state, axes, inverse=False):
    """Applies a controlled ternary add gate by slicing along the first axis specified in ``axes`` and
        applying a TShift transformation along the second axis. The ternary add gate acts on the computational
        basis states like :math:`	ext{TAdd}\x0bert i, j\rangle \rightarrow \x0bert i, i+j \rangle`, where addition
        is taken modulo 3.

        By slicing along the first axis, we are able to select all of the amplitudes with corresponding
        :math:`|1\rangle` and :math:`|2\rangle` for the control qutrit. This means we just need to apply
        a :class:`~.TShift` gate when slicing along index 1, and a :class:`~.TShift` adjoint gate when
        slicing along index 2

        Args:
            state (array[complex]): input state
            axes (List[int]): target axes to apply transformation

        Returns:
            array[complex]: output state
        """
    slices = [_get_slice(i, axes[0], self.num_wires) for i in range(3)]
    target_axes = [axes[1] - 1] if axes[1] > axes[0] else [axes[1]]
    state_1 = self._apply_tshift(state[slices[1]], axes=target_axes, inverse=inverse)
    state_2 = self._apply_tshift(state[slices[2]], axes=target_axes, inverse=not inverse)
    return self._stack([state[slices[0]], state_1, state_2], axis=axes[0])