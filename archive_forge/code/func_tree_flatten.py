from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union
import jax
import jax.numpy as jnp
from jax.experimental import sparse
from jax.tree_util import register_pytree_node_class
import pennylane as qml
from .parametrized_hamiltonian import ParametrizedHamiltonian
from .hardware_hamiltonian import HardwareHamiltonian
def tree_flatten(self):
    """Function used by ``jax`` to flatten the JaxLazyDot operation.

        Returns:
            tuple: tuple containing children and the auxiliary data of the class
        """
    children = (self.coeffs, self.mats)
    aux_data = None
    return (children, aux_data)