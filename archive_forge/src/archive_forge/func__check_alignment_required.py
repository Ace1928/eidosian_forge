from __future__ import annotations
import itertools
import warnings
from collections import defaultdict
from collections.abc import Iterable
from typing import Type
from qiskit.circuit.quantumcircuit import ClbitSpecifier, QubitSpecifier
from qiskit.circuit.delay import Delay
from qiskit.circuit.measure import Measure
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.utils.deprecation import deprecate_func
def _check_alignment_required(dag: DAGCircuit, alignment: int, instructions: Type | list[Type]) -> bool:
    """Check DAG nodes and return a boolean representing if instruction scheduling is necessary.

    Args:
        dag: DAG circuit to check.
        alignment: Instruction alignment condition.
        instructions: Target instructions.

    Returns:
        If instruction scheduling is necessary.
    """
    if not isinstance(instructions, list):
        instructions = [instructions]
    if alignment == 1:
        return False
    if all((len(dag.op_nodes(inst)) == 0 for inst in instructions)):
        return False
    for delay_node in dag.op_nodes(Delay):
        duration = delay_node.op.duration
        if isinstance(duration, ParameterExpression):
            warnings.warn(f'Parametrized delay with {repr(duration)} is found in circuit {dag.name}. This backend requires alignment={alignment}. Please make sure all assigned values are multiple values of the alignment.', UserWarning)
        elif duration % alignment != 0:
            return True
    return False