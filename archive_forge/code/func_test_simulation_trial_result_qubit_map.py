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
def test_simulation_trial_result_qubit_map():
    q = cirq.LineQubit.range(2)
    result = cirq.Simulator().simulate(cirq.Circuit([cirq.CZ(q[0], q[1])]))
    assert result.qubit_map == {q[0]: 0, q[1]: 1}
    result = cirq.DensityMatrixSimulator().simulate(cirq.Circuit([cirq.CZ(q[0], q[1])]))
    assert result.qubit_map == {q[0]: 0, q[1]: 1}