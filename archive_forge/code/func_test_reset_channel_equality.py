import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_reset_channel_equality():
    assert cirq.reset(cirq.LineQubit(0)).gate == cirq.ResetChannel()
    assert cirq.reset(cirq.LineQid(0, 3)).gate == cirq.ResetChannel(3)