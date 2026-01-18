from typing import Tuple, List
from unittest.mock import create_autospec
import cirq
import numpy as np
from pyquil.gates import MEASURE, RX, DECLARE, H, CNOT, I
from pyquil.quilbase import Pragma, Reset
from cirq_rigetti import circuit_transformers as transformers
def test_transform_cirq_circuit_to_pyquil_program(parametric_circuit_with_params: Tuple[cirq.Circuit, cirq.Linspace]) -> None:
    """test that a user can transform a `cirq.Circuit` to a `pyquil.Program`
    functionally.
    """
    parametric_circuit, param_resolvers = parametric_circuit_with_params
    circuit = cirq.protocols.resolve_parameters(parametric_circuit, param_resolvers[1])
    program, _ = transformers.default(circuit=circuit)
    assert RX(np.pi / 2, 0) in program.instructions, 'executable should contain an RX(pi) 0 instruction'
    assert DECLARE('m0') in program.instructions, 'executable should declare a read out bit'
    assert MEASURE(0, ('m0', 0)) in program.instructions, 'executable should measure the read out bit'