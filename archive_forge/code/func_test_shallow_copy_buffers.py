from typing import cast, Type
from unittest import mock
import numpy as np
import pytest
import cirq
def test_shallow_copy_buffers():
    args = cirq.StateVectorSimulationState(qubits=cirq.LineQubit.range(1), initial_state=0)
    copy = args.copy(deep_copy_buffers=False)
    assert copy.available_buffer is args.available_buffer