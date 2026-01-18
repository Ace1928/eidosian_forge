import math
from typing import Optional, Tuple
import cirq
import cirq_ft
import numpy as np
import pytest
from attr import frozen
from cirq._compat import cached_property
from cirq_ft.algos.mean_estimation.complex_phase_oracle import ComplexPhaseOracle
from cirq_ft.infra import bit_tools
from cirq_ft.infra import testing as cq_testing
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@pytest.mark.parametrize('bitsize', [2, 3, 4, 5])
@pytest.mark.parametrize('arctan_bitsize', [5, 6, 7])
@allow_deprecated_cirq_ft_use_in_tests
def test_phase_oracle(bitsize: int, arctan_bitsize: int):
    phase_oracle = ComplexPhaseOracle(ExampleSelect(bitsize), arctan_bitsize)
    g = cq_testing.GateHelper(phase_oracle)
    circuit = cirq.Circuit(cirq.H.on_each(*g.quregs['selection']))
    circuit += cirq.Circuit(cirq.decompose_once(g.operation))
    qubit_order = cirq.QubitOrder.explicit(g.quregs['selection'], fallback=cirq.QubitOrder.DEFAULT)
    result = cirq.Simulator(dtype=np.complex128).simulate(circuit, qubit_order=qubit_order)
    state_vector = result.final_state_vector
    state_vector = state_vector.reshape(2 ** bitsize, len(state_vector) // 2 ** bitsize)
    prepared_state = state_vector.sum(axis=1)
    for x in range(2 ** bitsize):
        output_val = -2 * np.arctan(x, dtype=np.double) / np.pi
        output_bits = [*bit_tools.iter_bits_fixed_point(np.abs(output_val), arctan_bitsize)]
        approx_val = np.sign(output_val) * math.fsum([b * (1 / 2 ** (1 + i)) for i, b in enumerate(output_bits)])
        assert math.isclose(output_val, approx_val, abs_tol=1 / 2 ** bitsize), output_bits
        y = np.exp(1j * approx_val * np.pi) / np.sqrt(2 ** bitsize)
        assert np.isclose(prepared_state[x], y)