from unittest.mock import Mock, MagicMock
import io
import numpy as np
import pytest
import cirq
import cirq.contrib.routing as ccr
from cirq.contrib.quantum_volume import CompilationResult
def test_sample_heavy_set():
    """Test that we correctly sample a circuit's heavy set"""
    sampler = Mock(spec=cirq.Simulator)
    result = cirq.ResultDict(params=cirq.ParamResolver({}), measurements={'mock': np.array([[0, 1], [1, 0], [1, 1], [0, 0]])})
    sampler.run = MagicMock(return_value=result)
    circuit = cirq.Circuit(cirq.measure(*cirq.LineQubit.range(2)))
    compilation_result = CompilationResult(circuit=circuit, mapping={}, parity_map={})
    probability = cirq.contrib.quantum_volume.sample_heavy_set(compilation_result, [1, 2, 3], sampler=sampler, repetitions=10)
    assert probability == 0.75