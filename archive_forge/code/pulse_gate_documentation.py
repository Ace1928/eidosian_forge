from typing import List, Union
from qiskit.circuit import Instruction as CircuitInst
from qiskit.pulse import Schedule, ScheduleBlock
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.transpiler.target import Target
from qiskit.transpiler.exceptions import TranspilerError
from .base_builder import CalibrationBuilder
Gets the calibrated schedule for the given instruction and qubits.

        Args:
            node_op: Target instruction object.
            qubits: Integer qubit indices to check.

        Returns:
            Return Schedule of target gate instruction.

        Raises:
            TranspilerError: When node is parameterized and calibration is raw schedule object.
        