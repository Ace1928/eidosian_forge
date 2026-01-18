from typing import Iterator
import pytest
import sympy
import cirq
from cirq.study import sweeps
from cirq_google.study import DeviceParameter
from cirq_google.api import v2
def test_sweep_with_list_sweep():
    ls = cirq.study.to_sweep([{'a': 1, 'b': 2}, {'a': 3, 'b': 4}])
    proto = v2.sweep_to_proto(ls)
    expected = v2.run_context_pb2.Sweep()
    expected.sweep_function.function_type = v2.run_context_pb2.SweepFunction.ZIP
    p1 = expected.sweep_function.sweeps.add()
    p1.single_sweep.parameter_key = 'a'
    p1.single_sweep.points.points.extend([1, 3])
    p2 = expected.sweep_function.sweeps.add()
    p2.single_sweep.parameter_key = 'b'
    p2.single_sweep.points.points.extend([2, 4])
    assert proto == expected