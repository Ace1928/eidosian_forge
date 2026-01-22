from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Tuple
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
@dataclass
class _MinimumPointState:
    __slots__ = ('dag', 'score', 'since')
    dag: DAGCircuit
    score: Tuple[float, ...]
    since: int