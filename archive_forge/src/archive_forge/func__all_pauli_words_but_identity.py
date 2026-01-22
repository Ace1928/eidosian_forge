import pennylane as qml
from pennylane.operation import Operation, AnyWires
from pennylane.ops import PauliRot
def _all_pauli_words_but_identity(num_wires):
    yield from (_tuple_to_word(idx_tuple) for idx_tuple in _n_k_gray_code(4, num_wires, start=1))