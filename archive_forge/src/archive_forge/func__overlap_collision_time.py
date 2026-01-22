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
def _overlap_collision_time(c1: Sequence['cirq.Moment'], c2: Sequence['cirq.Moment'], align: 'cirq.Alignment') -> int:
    seen_times: Dict['cirq.Qid', int] = {}
    if align == Alignment.LEFT:
        upper_bound = len(c1)
    elif align == Alignment.RIGHT:
        upper_bound = len(c2)
    elif align == Alignment.FIRST:
        upper_bound = min(len(c1), len(c2))
    else:
        raise NotImplementedError(f'Unrecognized alignment: {align}')
    t = 0
    while t < upper_bound:
        if t < len(c2):
            for op in c2[t]:
                for q in op.qubits:
                    k2 = seen_times.setdefault(q, t)
                    if k2 < 0:
                        upper_bound = min(upper_bound, t + ~k2)
        if t < len(c1):
            for op in c1[-1 - t]:
                for q in op.qubits:
                    k2 = seen_times.setdefault(q, ~t)
                    if k2 >= 0:
                        upper_bound = min(upper_bound, t + k2)
        t += 1
    return upper_bound