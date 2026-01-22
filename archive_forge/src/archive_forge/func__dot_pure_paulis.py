from collections import defaultdict
from typing import Sequence, Union, Callable
import pennylane as qml
from pennylane.operation import Operator, convert_to_opmath
from pennylane.pulse import ParametrizedHamiltonian
from pennylane.pauli import PauliWord, PauliSentence
def _dot_pure_paulis(coeffs: Sequence[float], ops: Sequence[Union[PauliWord, PauliSentence]]):
    """Faster computation of dot when all ops are PauliSentences or PauliWords"""
    return sum((c * op for c, op in zip(coeffs[1:], ops[1:])), start=coeffs[0] * ops[0])