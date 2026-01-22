import functools
import pennylane as qml
from pennylane.operation import Operation, AnyWires
@functools.lru_cache()
def _state_preparation_pauli_words(num_wires):
    """Pauli words necessary for a state preparation.

    Args:
        num_wires (int): Number of wires of the state preparation

    Returns:
        List[str]: List of all necessary Pauli words for the state preparation
    """
    if num_wires == 1:
        return ['X', 'Y']
    sub_pauli_words = _state_preparation_pauli_words(num_wires - 1)
    sub_id = 'I' * (num_wires - 1)
    single_qubit_words = ['X' + sub_id, 'Y' + sub_id]
    multi_qubit_words = list(map(lambda word: 'I' + word, sub_pauli_words)) + list(map(lambda word: 'X' + word, sub_pauli_words))
    return single_qubit_words + multi_qubit_words