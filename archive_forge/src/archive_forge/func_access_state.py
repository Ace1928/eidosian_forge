import abc
import itertools
import warnings
from collections import defaultdict
from typing import Union, List
import inspect
import logging
import numpy as np
import pennylane as qml
from pennylane import Device, DeviceError
from pennylane.math import multiply as qmlmul
from pennylane.math import sum as qmlsum
from pennylane.measurements import (
from pennylane.resource import Resources
from pennylane.operation import operation_derivative, Operation
from pennylane.tape import QuantumTape
from pennylane.wires import Wires
def access_state(self, wires=None):
    """Check that the device has access to an internal state and return it if available.

        Args:
            wires (Wires): wires of the reduced system

        Raises:
            QuantumFunctionError: if the device is not capable of returning the state

        Returns:
            array or tensor: the state or the density matrix of the device
        """
    if not self.capabilities().get('returns_state'):
        raise qml.QuantumFunctionError('The current device is not capable of returning the state')
    state = getattr(self, 'state', None)
    if state is None:
        raise qml.QuantumFunctionError('The state is not available in the current device')
    if wires:
        density_matrix = self.density_matrix(wires)
        return density_matrix
    return state