from typing import Iterator
import pytest
import sympy
import cirq
from cirq.study import sweeps
from cirq_google.study import DeviceParameter
from cirq_google.api import v2
def test_sweep_from_proto_unknown_sweep_type():
    with pytest.raises(ValueError, match='cannot convert to v2 Sweep proto'):
        v2.sweep_to_proto(UnknownSweep('foo'))