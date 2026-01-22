from __future__ import annotations
from typing import Optional, List, Tuple, Union, Iterable
import qiskit.circuit
from qiskit.circuit import Barrier, Delay
from qiskit.circuit import Instruction, ParameterExpression
from qiskit.circuit.duration import duration_in_dt
from qiskit.providers import Backend
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.utils.units import apply_prefix
def _convert_unit(self, duration: float, from_unit: str, to_unit: str) -> float:
    if from_unit.endswith('s') and from_unit != 's':
        duration = apply_prefix(duration, from_unit)
        from_unit = 's'
    if from_unit == to_unit:
        return duration
    if self.dt is None:
        raise TranspilerError(f"dt is necessary to convert durations from '{from_unit}' to '{to_unit}'")
    if from_unit == 's' and to_unit == 'dt':
        if isinstance(duration, ParameterExpression):
            return duration / self.dt
        return duration_in_dt(duration, self.dt)
    elif from_unit == 'dt' and to_unit == 's':
        return duration * self.dt
    else:
        raise TranspilerError(f"Conversion from '{from_unit}' to '{to_unit}' is not supported")