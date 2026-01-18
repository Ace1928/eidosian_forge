import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_phase_damping_channel_str():
    assert str(cirq.phase_damp(0.3)) == 'phase_damp(gamma=0.3)'