from typing import Iterator
import pytest
import sympy
import cirq
from cirq.study import sweeps
from cirq_google.study import DeviceParameter
from cirq_google.api import v2
@pytest.mark.parametrize('pass_out', [False, True])
def test_run_context_to_proto(pass_out: bool) -> None:
    msg = v2.run_context_pb2.RunContext() if pass_out else None
    out = v2.run_context_to_proto(None, 10, out=msg)
    if pass_out:
        assert out is msg
    assert len(out.parameter_sweeps) == 1
    assert v2.sweep_from_proto(out.parameter_sweeps[0].sweep) == cirq.UnitSweep
    assert out.parameter_sweeps[0].repetitions == 10
    sweep = cirq.Linspace('a', 0, 1, 21)
    msg = v2.run_context_pb2.RunContext() if pass_out else None
    out = v2.run_context_to_proto(sweep, 100, out=msg)
    if pass_out:
        assert out is msg
    assert len(out.parameter_sweeps) == 1
    assert v2.sweep_from_proto(out.parameter_sweeps[0].sweep) == sweep
    assert out.parameter_sweeps[0].repetitions == 100