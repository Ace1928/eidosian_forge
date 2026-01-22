import dataclasses
import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union
import numpy as np
import quimb.tensor as qtn
from cirq import devices, protocols, qis, value
from cirq.sim import simulator_base
from cirq.sim.simulation_state import SimulationState
class MPSSimulatorStepResult(simulator_base.StepResultBase['MPSState']):
    """A `StepResult` that can perform measurements."""

    def __init__(self, sim_state: 'cirq.SimulationStateBase[MPSState]'):
        """Results of a step of the simulator.
        Attributes:
            sim_state: The qubit:SimulationState lookup for this step.
        """
        super().__init__(sim_state)

    @property
    def state(self):
        return self._merged_sim_state

    def __str__(self) -> str:

        def bitstring(vals):
            return ','.join((str(v) for v in vals))
        results = sorted([(key, bitstring(val)) for key, val in self.measurements.items()])
        if len(results) == 0:
            measurements = ''
        else:
            measurements = ' '.join([f'{key}={val}' for key, val in results]) + '\n'
        final = self.state
        return f'{measurements}{final}'

    def _repr_pretty_(self, p: Any, cycle: bool):
        """iPython (Jupyter) pretty print."""
        p.text('cirq.MPSSimulatorStepResult(...)' if cycle else self.__str__())