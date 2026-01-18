from typing import Iterator
import pytest
import sympy
import cirq
from cirq.study import sweeps
from cirq_google.study import DeviceParameter
from cirq_google.api import v2
def test_sweep_from_proto_sweep_function_not_set():
    proto = v2.run_context_pb2.Sweep()
    proto.sweep_function.sweeps.add()
    with pytest.raises(ValueError, match='invalid sweep function type'):
        v2.sweep_from_proto(proto)