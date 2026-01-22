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
class FakeNonIterSimulatorImpl(SimulatesAmplitudes, SimulatesExpectationValues, SimulatesFinalState):
    """A class which defines the non-Iterator simulator API methods.

        After v0.12, simulators are expected to implement the *_iter methods.
        """

    def compute_amplitudes_sweep(self, program: 'cirq.AbstractCircuit', bitstrings: Sequence[int], params: study.Sweepable, qubit_order: cirq.QubitOrderOrList=cirq.QubitOrder.DEFAULT) -> Sequence[Sequence[complex]]:
        return [[1.0]]

    def simulate_expectation_values_sweep(self, program: 'cirq.AbstractCircuit', observables: Union['cirq.PauliSumLike', List['cirq.PauliSumLike']], params: 'study.Sweepable', qubit_order: cirq.QubitOrderOrList=cirq.QubitOrder.DEFAULT, initial_state: Any=None, permit_terminal_measurements: bool=False) -> List[List[float]]:
        return [[1.0]]

    def simulate_sweep(self, program: 'cirq.AbstractCircuit', params: study.Sweepable, qubit_order: cirq.QubitOrderOrList=cirq.QubitOrder.DEFAULT, initial_state: Any=None) -> List[SimulationTrialResult]:
        return [mock_trial_result]