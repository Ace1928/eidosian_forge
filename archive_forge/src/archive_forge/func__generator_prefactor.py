import inspect
import warnings
import numpy as np
import pennylane as qml
from pennylane.ops import Hamiltonian, SProd, Prod, Sum
def _generator_prefactor(gen):
    """Return the generator as ```(obs, prefactor)`` representing
    :math:`G=p \\hat{O}`, where

    - prefactor :math:`p` is a float
    - observable `\\hat{O}` is one of :class:`~.Hermitian`,
      :class:`~.SparseHamiltonian`, or a tensor product
      of Pauli words.
    """
    prefactor = 1.0
    if isinstance(gen, Prod):
        gen = qml.simplify(gen)
    if isinstance(gen, Hamiltonian):
        gen = qml.dot(gen.coeffs, gen.ops)
    if isinstance(gen, Sum):
        ops = [o.base if isinstance(o, SProd) else o for o in gen]
        coeffs = [o.scalar if isinstance(o, SProd) else 1 for o in gen]
        abs_coeffs = list(qml.math.abs(coeffs))
        if qml.math.allequal(coeffs[0], coeffs):
            return (qml.sum(*ops), coeffs[0])
        if qml.math.allequal(abs_coeffs[0], abs_coeffs):
            prefactor = abs_coeffs[0]
            coeffs = [c / prefactor for c in coeffs]
            return (qml.dot(coeffs, ops), prefactor)
    elif isinstance(gen, SProd):
        return (gen.base, gen.scalar)
    return (gen, prefactor)