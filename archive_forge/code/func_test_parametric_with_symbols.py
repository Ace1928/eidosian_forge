from typing import Tuple, Any, List, Union, Dict
import pytest
import cirq
from pyquil import Program
import numpy as np
import sympy
from cirq_rigetti import circuit_sweep_executors as executors, circuit_transformers
def test_parametric_with_symbols(mock_qpu_implementer: Any, parametric_circuit_with_params: Tuple[cirq.Circuit, cirq.Linspace]):
    parametric_circuit, _ = parametric_circuit_with_params
    repetitions = 2
    expected_results = [np.ones((repetitions,))]
    quantum_computer = mock_qpu_implementer.implement_passive_quantum_computer_with_results(expected_results)
    with pytest.raises(ValueError, match='Symbols not valid'):
        _ = executors.with_quilc_parametric_compilation(quantum_computer=quantum_computer, circuit=parametric_circuit, resolvers=[{sympy.Symbol('a') + sympy.Symbol('b'): sympy.Symbol('c')}], repetitions=repetitions)