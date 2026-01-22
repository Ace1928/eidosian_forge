import itertools
from typing import (
import networkx as nx
import rustworkx as rx
import numpy as np
import pennylane as qml
from pennylane.ops import Hamiltonian
def _square_hamiltonian_terms(coeffs: Iterable[float], ops: Iterable[qml.operation.Observable]) -> Tuple[List[float], List[qml.operation.Observable]]:
    """Calculates the coefficients and observables that compose the squared Hamiltonian.

    Args:
        coeffs (Iterable[float]): coeffients of the input Hamiltonian
        ops (Iterable[qml.operation.Observable]): observables of the input Hamiltonian

    Returns:
        Tuple[List[float], List[qml.operation.Observable]]: The list of coefficients and list of observables
        of the squared Hamiltonian.
    """
    squared_coeffs, squared_ops = ([], [])
    pairs = [(coeff, op) for coeff, op in zip(coeffs, ops)]
    products = itertools.product(pairs, repeat=2)
    for (coeff1, op1), (coeff2, op2) in products:
        squared_coeffs.append(coeff1 * coeff2)
        if isinstance(op1, qml.Identity):
            squared_ops.append(op2)
        elif isinstance(op2, qml.Identity):
            squared_ops.append(op1)
        elif op1.wires == op2.wires and isinstance(op1, type(op2)):
            squared_ops.append(qml.Identity(0))
        elif op2.wires[0] < op1.wires[0]:
            squared_ops.append(op2 @ op1)
        else:
            squared_ops.append(op1 @ op2)
    return (squared_coeffs, squared_ops)