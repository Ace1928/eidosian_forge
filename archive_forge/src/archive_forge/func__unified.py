from typing import Set
from qiskit.circuit import Delay
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.instruction_durations import InstructionDurations
from qiskit.transpiler.target import Target
@staticmethod
def _unified(unit_set: Set[str]) -> str:
    if not unit_set:
        return 'dt'
    if len(unit_set) == 1 and 'dt' in unit_set:
        return 'dt'
    all_si = True
    for unit in unit_set:
        if not unit.endswith('s'):
            all_si = False
            break
    if all_si:
        return 'SI'
    return 'mixed'