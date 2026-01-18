import tempfile
from typing import Iterable, Dict, List
import numpy as np
import pytest
import cirq
import cirq.work as cw
from cirq.work import _MeasurementSpec, BitstringAccumulator, group_settings_greedy, InitObsSetting
from cirq.work.observable_measurement import (
def test_with_parameterized_layers():
    qs = cirq.LineQubit.range(3)
    circuit = cirq.Circuit([cirq.H.on_each(*qs), cirq.CZ(qs[0], qs[1]), cirq.CZ(qs[1], qs[2])])
    circuit2 = _with_parameterized_layers(circuit, qubits=qs, needs_init_layer=False)
    assert circuit != circuit2
    assert len(circuit2) == 3 + 3
    *_, xlayer, ylayer, measurelayer = circuit2.moments
    for op in xlayer.operations:
        assert isinstance(op.gate, cirq.XPowGate)
        assert op.gate.exponent.name.endswith('-Xf')
    for op in ylayer.operations:
        assert isinstance(op.gate, cirq.YPowGate)
        assert op.gate.exponent.name.endswith('-Yf')
    for op in measurelayer:
        assert isinstance(op.gate, cirq.MeasurementGate)
    circuit3 = _with_parameterized_layers(circuit, qubits=qs, needs_init_layer=True)
    assert circuit != circuit3
    assert circuit2 != circuit3
    assert len(circuit3) == 2 + 3 + 3
    xlayer, ylayer, *_ = circuit3.moments
    for op in xlayer.operations:
        assert isinstance(op.gate, cirq.XPowGate)
        assert op.gate.exponent.name.endswith('-Xi')
    for op in ylayer.operations:
        assert isinstance(op.gate, cirq.YPowGate)
        assert op.gate.exponent.name.endswith('-Yi')