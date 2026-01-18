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
def test_trial_result_initializer():
    resolver = cirq.ParamResolver()
    state = 3
    x = SimulationTrialResult(resolver, {}, state)
    assert x._final_simulator_state == 3
    x = SimulationTrialResult(resolver, {}, final_simulator_state=state)
    assert x._final_simulator_state == 3