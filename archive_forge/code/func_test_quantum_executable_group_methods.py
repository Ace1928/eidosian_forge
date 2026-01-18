import dataclasses
import cirq
import cirq_google
import pytest
from cirq_google import (
def test_quantum_executable_group_methods():
    exes = _get_quantum_executables()
    eg = QuantumExecutableGroup(exes)
    assert str(eg) == "QuantumExecutableGroup(executables=[QuantumExecutable(spec=cirq_google.KeyValueExecutableSpec(executable_family='cirq_google.algo_benchmarks.example', key_value_pairs=(('name', 'example-program-0'),))), QuantumExecutable(spec=cirq_google.KeyValueExecutableSpec(executable_family='cirq_google.algo_benchmarks.example', key_value_pairs=(('name', 'example-program-1'),))), ...])"
    assert len(eg) == len(exes), '__len__'
    assert exes == [e for e in eg], '__iter__'