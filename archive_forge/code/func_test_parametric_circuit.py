from typing import Any, List, Sequence, Tuple
import cirq
import pytest
from pyquil import Program
from pyquil.api import QuantumComputer
import numpy as np
from pyquil.gates import MEASURE, RX, X, DECLARE, H, CNOT
from cirq_rigetti import RigettiQCSService
from typing_extensions import Protocol
from cirq_rigetti import circuit_transformers as transformers
from cirq_rigetti import circuit_sweep_executors as executors
@pytest.mark.parametrize('result_builder', [_build_service_results, _build_sampler_results])
def test_parametric_circuit(mock_qpu_implementer: Any, parametric_circuit_with_params: Tuple[cirq.Circuit, cirq.Sweepable], result_builder: _ResultBuilder) -> None:
    """test that RigettiQCSService and RigettiQCSSampler can run a parametric
    circuit with a specified set of parameters and return expected cirq.Results.
    """
    parametric_circuit = parametric_circuit_with_params[0]
    sweepable = parametric_circuit_with_params[1]
    results, quantum_computer, expected_results, param_resolvers = result_builder(mock_qpu_implementer, parametric_circuit, sweepable)
    assert len(param_resolvers) == len(results), 'should return a result for every element in sweepable'
    for i, param_resolver in enumerate(param_resolvers):
        result = results[i]
        assert param_resolver == result.params
        assert np.allclose(result.measurements['m'], expected_results[i]), 'should return an ordered list of results with correct set of measurements'

    def test_executable(i: int, program: Program) -> None:
        params = param_resolvers[i]
        t = params['t']
        if t == 1:
            assert X(0) in program.instructions, f'executable should contain an X(0) instruction at {i}'
        else:
            assert RX(np.pi * t, 0) in program.instructions, f'executable should contain an RX(pi*{t}) 0 instruction at {i}'
        assert DECLARE('m0') in program.instructions, 'executable should declare a read out bit'
        assert MEASURE(0, ('m0', 0)) in program.instructions, 'executable should measure the read out bit'
    param_sweeps = len(param_resolvers)
    assert param_sweeps == quantum_computer.compiler.quil_to_native_quil.call_count
    for i, call_args in enumerate(quantum_computer.compiler.quil_to_native_quil.call_args_list):
        test_executable(i, call_args[0][0])
    assert param_sweeps == quantum_computer.compiler.native_quil_to_executable.call_count
    for i, call_args in enumerate(quantum_computer.compiler.native_quil_to_executable.call_args_list):
        test_executable(i, call_args[0][0])
    assert param_sweeps == quantum_computer.qam.run.call_count
    for i, call_args in enumerate(quantum_computer.qam.run.call_args_list):
        test_executable(i, call_args[0][0])