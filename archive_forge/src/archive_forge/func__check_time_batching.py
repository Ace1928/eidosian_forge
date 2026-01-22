from typing import List, Union, Sequence
import warnings
import pennylane as qml
from pennylane.operation import AnyWires, Operation
from pennylane.typing import TensorLike
from pennylane.ops import functions
from .parametrized_hamiltonian import ParametrizedHamiltonian
from .hardware_hamiltonian import HardwareHamiltonian
def _check_time_batching(self):
    """Check whether the time argument is broadcasted/batched."""
    if not self.hyperparameters['return_intermediate'] or self.t is None:
        return
    self._batch_size = self.t.shape[0]