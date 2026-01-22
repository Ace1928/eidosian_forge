from typing import List, Sequence
import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft import infra
from cirq_ft.infra.bit_tools import iter_bits
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
def _fake_prepare(positive_coefficients: np.ndarray, selection_register: List[cirq.Qid]) -> cirq.OP_TREE:
    pos_coeffs = positive_coefficients.flatten()
    size_hilbert_of_reg = 2 ** len(selection_register)
    assert len(pos_coeffs) <= size_hilbert_of_reg
    if len(pos_coeffs) < size_hilbert_of_reg:
        pos_coeffs = np.hstack((pos_coeffs, np.array([0] * (size_hilbert_of_reg - len(pos_coeffs)))))
    assert np.isclose(pos_coeffs.conj().T @ pos_coeffs, 1.0)
    circuit = cirq.Circuit()
    circuit.append(cirq.StatePreparationChannel(pos_coeffs).on(*selection_register))
    return circuit