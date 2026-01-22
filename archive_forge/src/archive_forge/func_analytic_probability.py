import functools
import numpy as np
import pennylane as qml  # pylint: disable=unused-import
from pennylane import QutritDevice, QutritBasisState, DeviceError
from pennylane.wires import WireError
from pennylane.devices.default_qubit_legacy import _get_slice
from .._version import __version__
def analytic_probability(self, wires=None):
    if self._state is None:
        return None
    flat_state = self._flatten(self._state)
    real_state = self._real(flat_state)
    imag_state = self._imag(flat_state)
    prob = self.marginal_prob(real_state ** 2 + imag_state ** 2, wires)
    return prob