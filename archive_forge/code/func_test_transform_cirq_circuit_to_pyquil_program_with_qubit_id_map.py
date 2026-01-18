from typing import Tuple, List
from unittest.mock import create_autospec
import cirq
import numpy as np
from pyquil.gates import MEASURE, RX, DECLARE, H, CNOT, I
from pyquil.quilbase import Pragma, Reset
from cirq_rigetti import circuit_transformers as transformers
def test_transform_cirq_circuit_to_pyquil_program_with_qubit_id_map(bell_circuit_with_qids: Tuple[cirq.Circuit, List[cirq.Qid]]) -> None:
    """test that a user can transform a `cirq.Circuit` to a `pyquil.Program`
    functionally with explicit physical qubit address mapping.
    """
    bell_circuit, qubits = bell_circuit_with_qids
    qubit_id_map = {qubits[1]: '11', qubits[0]: '13'}
    transformer = transformers.build(qubit_id_map=qubit_id_map)
    program, _ = transformer(circuit=bell_circuit)
    assert H(13) in program.instructions, 'bell circuit should include Hadamard'
    assert CNOT(13, 11) in program.instructions, 'bell circuit should include CNOT'
    assert DECLARE('m0', memory_size=2) in program.instructions, 'executable should declare a read out bit'
    assert MEASURE(13, ('m0', 0)) in program.instructions, 'executable should measure the first qubit to the first read out bit'
    assert MEASURE(11, ('m0', 1)) in program.instructions, 'executable should measure the second qubit to the second read out bit'