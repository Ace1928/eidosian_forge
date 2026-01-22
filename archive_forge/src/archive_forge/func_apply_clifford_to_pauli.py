from typing import List, Optional, Sequence, cast
from qcs_api_client.client import QCSClientConfiguration
from pyquil.api._abstract_compiler import AbstractBenchmarker
from pyquil.api._compiler_client import (
from pyquil.paulis import PauliTerm, is_identity
from pyquil.quil import address_qubits, Program
from pyquil.quilbase import Gate
def apply_clifford_to_pauli(self, clifford: Program, pauli_in: PauliTerm) -> PauliTerm:
    """
        Given a circuit that consists only of elements of the Clifford group,
        return its action on a PauliTerm.

        In particular, for Clifford C, and Pauli P, this returns the PauliTerm
        representing CPC^{\\dagger}.

        :param clifford: A Program that consists only of Clifford operations.
        :param pauli_in: A PauliTerm to be acted on by clifford via conjugation.
        :return: A PauliTerm corresponding to clifford * pauli_in * clifford^{\\dagger}
        """
    if is_identity(pauli_in):
        return pauli_in
    indices_and_terms = list(zip(*list(pauli_in.operations_as_set())))
    request = ConjugatePauliByCliffordRequest(pauli_indices=list(indices_and_terms[0]), pauli_symbols=list(indices_and_terms[1]), clifford=clifford.out(calibrations=False))
    response = self._compiler_client.conjugate_pauli_by_clifford(request)
    phase_factor, paulis = (response.phase_factor, response.pauli)
    pauli_out = PauliTerm('I', 0, 1j ** phase_factor)
    clifford_qubits = clifford.get_qubits()
    pauli_qubits = pauli_in.get_qubits()
    all_qubits = sorted(set(cast(List[int], pauli_qubits)).union(set(cast(List[int], clifford_qubits))))
    for i, pauli in enumerate(paulis):
        pauli_out = cast(PauliTerm, pauli_out * PauliTerm(pauli, all_qubits[i]))
    return cast(PauliTerm, pauli_out * pauli_in.coefficient)