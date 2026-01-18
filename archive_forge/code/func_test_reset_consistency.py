import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_reset_consistency():
    two_d_chan = cirq.ResetChannel()
    cirq.testing.assert_has_consistent_apply_channel(two_d_chan)
    three_d_chan = cirq.ResetChannel(dimension=3)
    cirq.testing.assert_has_consistent_apply_channel(three_d_chan)