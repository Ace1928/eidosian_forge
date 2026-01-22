import numpy as np
import pennylane as qml
from pennylane.pauli.utils import is_pauli_word, pauli_to_binary, _wire_map_from_pauli_pair
def _create_commute_function():
    """This function constructs the ``_commutes`` helper utility function while using closure
    to hide the ``commutation_map`` data away from the global scope of the file.
    This function only needs to be called a single time.
    Returns:
        function
    """
    pauliz_group = {'PauliZ', 'ctrl', 'S', 'Adjoint(S)', 'T', 'Adjoint(T)', 'RZ', 'PhaseShift', 'MultiRZ', 'Identity', 'U1', 'IsingZZ'}
    swap_group = {'SWAP', 'ISWAP', 'SISWAP', 'Identity', 'Adjoint(ISWAP)', 'Adjoint(SISWAP)'}
    paulix_group = {'PauliX', 'SX', 'RX', 'Identity', 'IsingXX', 'Adjoint(SX)'}
    pauliy_group = {'PauliY', 'RY', 'Identity', 'IsingYY'}
    commutation_map = {}
    for group in [paulix_group, pauliy_group, pauliz_group, swap_group]:
        for op in group:
            commutation_map[op] = group
    identity_only = {'Hadamard', 'U2', 'U3', 'Rot'}
    for op in identity_only:
        commutation_map[op] = {'Identity', op}
    commutation_map['Identity'] = pauliz_group.union(swap_group, paulix_group, pauliy_group, identity_only)

    def commutes_inner(op_name1, op_name2):
        """Determine whether or not two operations commute.

        Relies on ``commutation_map`` from the enclosing namespace of ``_create_commute_function``.

        Args:
            op_name1 (str): name of one operation
            op_name2 (str): name of the second operation

        Returns:
            Bool

        """
        return op_name1 in commutation_map[op_name2]
    return commutes_inner