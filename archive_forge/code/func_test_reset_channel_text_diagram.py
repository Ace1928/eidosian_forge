import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_reset_channel_text_diagram():
    assert cirq.circuit_diagram_info(cirq.ResetChannel()) == cirq.CircuitDiagramInfo(wire_symbols=('R',))
    assert cirq.circuit_diagram_info(cirq.ResetChannel(3)) == cirq.CircuitDiagramInfo(wire_symbols=('R',))