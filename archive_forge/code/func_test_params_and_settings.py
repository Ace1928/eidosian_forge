import tempfile
from typing import Iterable, Dict, List
import numpy as np
import pytest
import cirq
import cirq.work as cw
from cirq.work import _MeasurementSpec, BitstringAccumulator, group_settings_greedy, InitObsSetting
from cirq.work.observable_measurement import (
def test_params_and_settings():
    qubits = cirq.LineQubit.range(1)
    q, = qubits
    tests = [(cirq.KET_ZERO, cirq.Z, 1), (cirq.KET_ONE, cirq.Z, -1), (cirq.KET_PLUS, cirq.X, 1), (cirq.KET_MINUS, cirq.X, -1), (cirq.KET_IMAG, cirq.Y, 1), (cirq.KET_MINUS_IMAG, cirq.Y, -1), (cirq.KET_ZERO, cirq.Y, 0)]
    for init, obs, coef in tests:
        setting = cw.InitObsSetting(init_state=init(q), observable=obs(q))
        circuit = cirq.Circuit(cirq.I.on_each(*qubits))
        circuit = _with_parameterized_layers(circuit, qubits=qubits, needs_init_layer=True)
        params = _get_params_for_setting(setting, flips=[False], qubits=qubits, needs_init_layer=True)
        circuit = circuit[:-1]
        psi = cirq.Simulator().simulate(circuit, param_resolver=params)
        z = cirq.Z(q).expectation_from_state_vector(psi.final_state_vector, qubit_map=psi.qubit_map)
        assert np.abs(coef - z) < 0.01, f'{init} {obs} {coef}'