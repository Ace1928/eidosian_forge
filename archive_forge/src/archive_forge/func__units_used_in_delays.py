from typing import Set
from qiskit.circuit import Delay
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.instruction_durations import InstructionDurations
from qiskit.transpiler.target import Target
@staticmethod
def _units_used_in_delays(dag: DAGCircuit) -> Set[str]:
    units_used = set()
    for node in dag.op_nodes(op=Delay):
        units_used.add(node.op.unit)
    return units_used