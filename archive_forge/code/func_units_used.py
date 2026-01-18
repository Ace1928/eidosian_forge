from __future__ import annotations
from typing import Optional, List, Tuple, Union, Iterable
import qiskit.circuit
from qiskit.circuit import Barrier, Delay
from qiskit.circuit import Instruction, ParameterExpression
from qiskit.circuit.duration import duration_in_dt
from qiskit.providers import Backend
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.utils.units import apply_prefix
def units_used(self) -> set[str]:
    """Get the set of all units used in this instruction durations.

        Returns:
            Set of units used in this instruction durations.
        """
    units_used = set()
    for _, unit in self.duration_by_name_qubits.values():
        units_used.add(unit)
    for _, unit in self.duration_by_name.values():
        units_used.add(unit)
    return units_used