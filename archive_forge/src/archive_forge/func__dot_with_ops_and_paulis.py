from collections import defaultdict
from typing import Sequence, Union, Callable
import pennylane as qml
from pennylane.operation import Operator, convert_to_opmath
from pennylane.pulse import ParametrizedHamiltonian
from pennylane.pauli import PauliWord, PauliSentence
def _dot_with_ops_and_paulis(coeffs: Sequence[float], ops: Sequence[Operator]):
    """Compute dot when operators are a mix of pennylane operators, PauliWord and PauliSentence by turning them all into a PauliSentence instance.
    Returns a PauliSentence instance"""
    pauli_words = defaultdict(lambda: 0)
    for coeff, op in zip(coeffs, ops):
        sentence = qml.pauli.pauli_sentence(op)
        for pw in sentence:
            pauli_words[pw] += sentence[pw] * coeff
    return qml.pauli.PauliSentence(pauli_words)