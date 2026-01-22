import dataclasses
import numbers
from typing import (
import sympy
from cirq import ops, value, protocols
def _max_weight_observable(observables: Iterable[ops.PauliString]) -> Optional[ops.PauliString]:
    """Create a new observable that is compatible with all input observables
    and has the maximum non-identity elements.

    The returned PauliString is constructed by taking the non-identity
    single-qubit Pauli at each qubit position.

    This function will return `None` if the input observables do not share a
    tensor product basis.

    For example, the _max_weight_observable of ["XI", "IZ"] is "XZ". Asking for
    the max weight observable of something like ["XI", "ZI"] will return None.

    The returned value need not actually be present in the input observables.
    Coefficients from input observables will be dropped.
    """
    qubit_pauli_map: Dict[ops.Qid, ops.Pauli] = {}
    for observable in observables:
        for qubit, pauli in observable.items():
            if qubit in qubit_pauli_map:
                if qubit_pauli_map[qubit] != pauli:
                    return None
            else:
                qubit_pauli_map[qubit] = pauli
    return ops.PauliString(qubit_pauli_map)