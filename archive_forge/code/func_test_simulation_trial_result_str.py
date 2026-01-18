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
def test_simulation_trial_result_str():
    assert str(cirq.SimulationTrialResult(params=cirq.ParamResolver({'s': 1}), measurements={}, final_simulator_state=(0, 1))) == '(no measurements)'
    assert str(cirq.SimulationTrialResult(params=cirq.ParamResolver({'s': 1}), measurements={'m': np.array([1])}, final_simulator_state=(0, 1))) == 'm=1'
    assert str(cirq.SimulationTrialResult(params=cirq.ParamResolver({'s': 1}), measurements={'m': np.array([1, 2, 3])}, final_simulator_state=(0, 1))) == 'm=123'
    assert str(cirq.SimulationTrialResult(params=cirq.ParamResolver({'s': 1}), measurements={'m': np.array([9, 10, 11])}, final_simulator_state=(0, 1))) == 'm=9 10 11'