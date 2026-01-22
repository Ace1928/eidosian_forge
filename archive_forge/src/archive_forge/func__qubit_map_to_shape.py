import abc
import collections
from typing import (
import numpy as np
from cirq import circuits, ops, protocols, study, value, work
from cirq.sim.simulation_state_base import SimulationStateBase
def _qubit_map_to_shape(qubit_map: Mapping['cirq.Qid', int]) -> Tuple[int, ...]:
    qid_shape: List[int] = [-1] * len(qubit_map)
    try:
        for q, i in qubit_map.items():
            qid_shape[i] = q.dimension
    except IndexError:
        raise ValueError(f'Invalid qubit_map. Qubit index out of bounds. Map is <{qubit_map!r}>.')
    if -1 in qid_shape:
        raise ValueError(f'Invalid qubit_map. Duplicate qubit index. Map is <{qubit_map!r}>.')
    return tuple(qid_shape)