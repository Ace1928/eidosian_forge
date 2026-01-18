from unittest.mock import Mock, MagicMock
import io
import numpy as np
import pytest
import cirq
import cirq.contrib.routing as ccr
from cirq.contrib.quantum_volume import CompilationResult
def test_sample_heavy_set_with_parity():
    """Test that we correctly sample a circuit's heavy set with a parity map"""
    sampler = Mock(spec=cirq.Simulator)
    result = cirq.ResultDict(params=cirq.ParamResolver({}), measurements={'q(0)': np.array([[1], [0]]), 'q(1)': np.array([[0], [1]]), 'q(2)': np.array([[1], [1]]), 'q(3)': np.array([[0], [0]])})
    sampler.run = MagicMock(return_value=result)
    circuit = cirq.Circuit(cirq.measure(*cirq.LineQubit.range(4)))
    compilation_result = CompilationResult(circuit=circuit, mapping={q: q for q in cirq.LineQubit.range(4)}, parity_map={cirq.LineQubit(0): cirq.LineQubit(1), cirq.LineQubit(2): cirq.LineQubit(3)})
    probability = cirq.contrib.quantum_volume.sample_heavy_set(compilation_result, [1], sampler=sampler, repetitions=1)
    assert probability == 0.5