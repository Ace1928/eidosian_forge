import abc
from typing import Generic, Dict, Any, List, Sequence, Union
from unittest import mock
import duet
import numpy as np
import pytest
import cirq
from cirq import study
from cirq.sim.simulation_state import TSimulationState
from cirq.sim.simulator import (
def test_run_simulator_run():
    expected_records = {'a': np.array([[[1]]])}
    simulator = FakeSimulatesSamples(expected_records)
    circuit = cirq.Circuit(cirq.measure(cirq.LineQubit(0), key='k'))
    param_resolver = cirq.ParamResolver({})
    expected_result = cirq.ResultDict(records=expected_records, params=param_resolver)
    assert expected_result == simulator.run(program=circuit, repetitions=10, param_resolver=param_resolver)