from typing import List
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.pulse.utils import format_meas_map
class ScheduleConfig:
    """Configuration for pulse scheduling."""

    def __init__(self, inst_map: InstructionScheduleMap, meas_map: List[List[int]], dt: float):
        """
        Container for information needed to schedule a QuantumCircuit into a pulse Schedule.

        Args:
            inst_map: The schedule definition of all gates supported on a backend.
            meas_map: A list of groups of qubits which have to be measured together.
            dt: Sample duration.
        """
        self.inst_map = inst_map
        self.meas_map = format_meas_map(meas_map)
        self.dt = dt