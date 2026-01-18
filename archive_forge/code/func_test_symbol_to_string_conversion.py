from typing import Iterator
import pytest
import sympy
import cirq
from cirq.study import sweeps
from cirq_google.study import DeviceParameter
from cirq_google.api import v2
def test_symbol_to_string_conversion():
    sweep = cirq.ListSweep([cirq.ParamResolver({sympy.Symbol('a'): 4.0})])
    proto = v2.sweep_to_proto(sweep)
    assert isinstance(proto, v2.run_context_pb2.Sweep)
    expected = v2.run_context_pb2.Sweep()
    expected.sweep_function.function_type = v2.run_context_pb2.SweepFunction.ZIP
    p1 = expected.sweep_function.sweeps.add()
    p1.single_sweep.parameter_key = 'a'
    p1.single_sweep.points.points.extend([4.0])
    assert proto == expected