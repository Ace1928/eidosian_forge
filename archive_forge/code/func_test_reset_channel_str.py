import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_reset_channel_str():
    assert str(cirq.ResetChannel()) == 'reset'
    assert str(cirq.ResetChannel(3)) == 'reset'