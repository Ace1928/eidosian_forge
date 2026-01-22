from __future__ import annotations
import functools
import warnings
from collections import defaultdict
from collections.abc import Iterable, Callable
from qiskit import circuit
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.pulse.calibration_entries import (
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.schedule import Schedule, ScheduleBlock
def _get_calibration_entry(self, instruction: str | circuit.instruction.Instruction, qubits: int | Iterable[int]) -> CalibrationEntry:
    """Return the :class:`.CalibrationEntry` without generating schedule.

        When calibration entry is un-parsed Pulse Qobj, this returns calibration
        without parsing it. :meth:`CalibrationEntry.get_schedule` method
        must be manually called with assigned parameters to get corresponding pulse schedule.

        This method is expected be directly used internally by the V2 backend converter
        for faster loading of the backend calibrations.

        Args:
            instruction: Name of the instruction or the instruction itself.
            qubits: The qubits for the instruction.

        Returns:
            The calibration entry.
        """
    instruction = _get_instruction_string(instruction)
    self.assert_has(instruction, qubits)
    return self._map[instruction][_to_tuple(qubits)]