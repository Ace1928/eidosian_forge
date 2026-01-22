from typing import List, Union, Sequence
import warnings
import pennylane as qml
from pennylane.operation import AnyWires, Operation
from pennylane.typing import TensorLike
from pennylane.ops import functions
from .parametrized_hamiltonian import ParametrizedHamiltonian
from .hardware_hamiltonian import HardwareHamiltonian
@functions.bind_new_parameters.register
def _bind_new_parameters_parametrized_evol(op: ParametrizedEvolution, params: Sequence[TensorLike]):
    return ParametrizedEvolution(op.H, params=params, t=op.t, return_intermediate=op.hyperparameters['return_intermediate'], complementary=op.hyperparameters['complementary'], dense=op.dense, **op.odeint_kwargs)