from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union
import jax
import jax.numpy as jnp
from jax.experimental import sparse
from jax.tree_util import register_pytree_node_class
import pennylane as qml
from .parametrized_hamiltonian import ParametrizedHamiltonian
from .hardware_hamiltonian import HardwareHamiltonian
Function used by ``jax`` to unflatten the ``JaxLazyDot`` pytree.

        Args:
            aux_data (None): empty argument
            children (tuple): tuple containing the coefficients and the matrices of the operation

        Returns:
            JaxLazyDot: JaxLazyDot instance
        