import dataclasses
import cirq
import cirq_google
import pytest
from cirq_google import (
def test_quantum_executable_group_to_tuple():
    exes1 = list(_get_quantum_executables())
    exes2 = tuple(_get_quantum_executables())
    eg1 = QuantumExecutableGroup(exes1)
    eg2 = QuantumExecutableGroup(exes2)
    assert hash(eg1) == hash(eg2)
    assert eg1 == eg2