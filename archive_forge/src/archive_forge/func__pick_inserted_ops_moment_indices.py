import abc
import enum
import html
import itertools
import math
from collections import defaultdict
from typing import (
from typing_extensions import Self
import networkx
import numpy as np
import cirq._version
from cirq import _compat, devices, ops, protocols, qis
from cirq._doc import document
from cirq.circuits._bucket_priority_queue import BucketPriorityQueue
from cirq.circuits.circuit_operation import CircuitOperation
from cirq.circuits.insert_strategy import InsertStrategy
from cirq.circuits.qasm_output import QasmOutput
from cirq.circuits.text_diagram_drawer import TextDiagramDrawer
from cirq.circuits.moment import Moment
from cirq.protocols import circuit_diagram_info_protocol
from cirq.type_workarounds import NotImplementedType
def _pick_inserted_ops_moment_indices(operations: Sequence['cirq.Operation'], start: int=0, frontier: Optional[Dict['cirq.Qid', int]]=None) -> Tuple[Sequence[int], Dict['cirq.Qid', int]]:
    """Greedily assigns operations to moments.

    Args:
        operations: The operations to assign to moments.
        start: The first moment to consider assignment to.
        frontier: The first moment to which an operation acting on a qubit
            can be assigned. Updated in place as operations are assigned.

    Returns:
        The frontier giving the index of the moment after the last one to
        which an operation that acts on each qubit is assigned. If a
        frontier was specified as an argument, this is the same object.
    """
    if frontier is None:
        frontier = defaultdict(lambda: 0)
    moment_indices = []
    for op in operations:
        op_start = max(start, max((frontier[q] for q in op.qubits), default=0))
        moment_indices.append(op_start)
        for q in op.qubits:
            frontier[q] = max(frontier[q], op_start + 1)
    return (moment_indices, frontier)