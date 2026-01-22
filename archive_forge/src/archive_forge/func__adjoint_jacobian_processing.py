from itertools import islice, product
from typing import List
import numpy as np
import pennylane as qml
from pennylane import BasisState, QubitDevice, StatePrep
from pennylane.devices import DefaultQubitLegacy
from pennylane.measurements import MeasurementProcess
from pennylane.operation import Operation
from pennylane.wires import Wires
from ._serialize import QuantumScriptSerializer
from ._version import __version__
@staticmethod
def _adjoint_jacobian_processing(jac):
    """
        Post-process the Jacobian matrix returned by ``adjoint_jacobian`` for
        the new return type system.
        """
    jac = np.squeeze(jac)
    if jac.ndim == 0:
        return np.array(jac)
    if jac.ndim == 1:
        return tuple((np.array(j) for j in jac))
    return tuple((tuple((np.array(j_) for j_ in j)) for j in jac))