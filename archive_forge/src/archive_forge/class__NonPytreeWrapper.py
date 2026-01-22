import logging
from typing import Tuple, Callable
import dataclasses
import jax
import jax.numpy as jnp
import pennylane as qml
from pennylane.transforms import convert_to_numpy_parameters
from pennylane.typing import ResultBatch
@dataclasses.dataclass
class _NonPytreeWrapper:
    """We aren't quite ready to switch to having tapes as pytrees as our
    differentiable argument due to:

    * Operators that aren't valid pytrees: ex. ParametrizedEvolution, ParametrizedHamiltonian, HardwareHamiltonian
    * Validation checks on initialization: see BasisStateProjector, StatePrep that does not allow the operator to store the cotangents
    * Jitting non-jax parametrized circuits.  NumPy parameters turn into abstract parameters during the pytree process.

    ``jax.custom_vjp`` forbids any non-differentiable argument to be a pytree, so we need to wrap it in a non-pytree type.

    When the above issues are fixed, we can treat ``tapes`` as the differentiable argument.

    """
    vals: Batch = None