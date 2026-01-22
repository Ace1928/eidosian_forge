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
class FakeStepResult(cirq.StepResult):

    def __init__(self, *, ones_qubits=None, final_state=None):
        self._ones_qubits = set(ones_qubits or [])
        self._final_state = final_state

    def _simulator_state(self):
        return self._final_state

    def state_vector(self):
        pass

    def __setstate__(self, state):
        pass

    def sample(self, qubits, repetitions=1, seed=None):
        return np.array([[qubit in self._ones_qubits for qubit in qubits]] * repetitions)