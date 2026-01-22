import functools
import itertools
from typing import Any, Dict, Generator, List, Sequence, Tuple
import sympy.parsing.sympy_parser as sympy_parser
import cirq
from cirq import value
from cirq.ops import raw_types
from cirq.ops.linear_combinations import PauliSum, PauliString
def _get_gates_from_hamiltonians(hamiltonian_polynomial_list: List['cirq.PauliSum'], qubit_map: Dict[str, 'cirq.Qid'], theta: float) -> Generator['cirq.Operation', None, None]:
    """Builds a circuit according to [1].

    Args:
        hamiltonian_polynomial_list: the list of Hamiltonians, typically built by calling
            PauliSum.from_boolean_expression().
        qubit_map: map of string (boolean variable name) to qubit.
        theta: A single float scaling the rotations.
    Yields:
        Gates that are the decomposition of the Hamiltonian.
    """
    combined = sum(hamiltonian_polynomial_list, PauliSum.from_pauli_strings(PauliString({})))
    qubit_names = sorted(qubit_map.keys())
    qubits = [qubit_map[name] for name in qubit_names]
    qubit_indices = {qubit: i for i, qubit in enumerate(qubits)}
    hamiltonians = {}
    for pauli_string in combined:
        w = pauli_string.coefficient.real
        qubit_idx = tuple(sorted((qubit_indices[qubit] for qubit in pauli_string.qubits)))
        hamiltonians[qubit_idx] = w

    def _apply_cnots(prevh: Tuple[int, ...], currh: Tuple[int, ...]):
        cnots: List[Tuple[int, int]] = []
        cnots.extend(((prevh[i], prevh[-1]) for i in range(len(prevh) - 1)))
        cnots.extend(((currh[i], currh[-1]) for i in range(len(currh) - 1)))
        cnots = _simplify_cnots(cnots)
        for gate in (cirq.CNOT(qubits[c], qubits[t]) for c, t in cnots):
            yield gate
    sorted_hamiltonian_keys = sorted(hamiltonians.keys(), key=functools.cmp_to_key(_gray_code_comparator))
    previous_h: Tuple[int, ...] = ()
    for h in sorted_hamiltonian_keys:
        w = hamiltonians[h]
        yield _apply_cnots(previous_h, h)
        if len(h) >= 1:
            yield cirq.Rz(rads=theta * w).on(qubits[h[-1]])
        previous_h = h
    yield _apply_cnots(previous_h, ())