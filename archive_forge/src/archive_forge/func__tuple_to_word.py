import pennylane as qml
from pennylane.operation import Operation, AnyWires
from pennylane.ops import PauliRot
def _tuple_to_word(index_tuple):
    """Convert an integer tuple to the corresponding Pauli word.

    The Pauli operators are converted as ``0 -> I``, ``1 -> X``,
    ``2 -> Y``, ``3 -> Z``.

    Args:
        index_tuple (Tuple[int]): An integer tuple describing the Pauli word

    Returns:
        str: The corresponding Pauli word
    """
    return ''.join([_PAULIS[i] for i in index_tuple])