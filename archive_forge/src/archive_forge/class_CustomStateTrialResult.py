from typing import Any, Dict, Generic, Sequence, Type, TYPE_CHECKING
import numpy as np
from cirq import sim
from cirq.sim.simulation_state import TSimulationState
class CustomStateTrialResult(sim.SimulationTrialResultBase[TSimulationState], Generic[TSimulationState]):
    """The trial result provided by `CustomStateSimulator.simulate`."""