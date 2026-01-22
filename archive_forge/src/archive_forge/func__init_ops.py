import json
import urllib.parse
from typing import (
import numpy as np
from cirq import devices, circuits, ops, protocols
from cirq.interop.quirk.cells import (
from cirq.interop.quirk.cells.parse import parse_matrix
def _init_ops(data: Dict[str, Any]) -> 'cirq.OP_TREE':
    if 'init' not in data:
        return []
    init = data['init']
    if not isinstance(init, List):
        raise ValueError(f'Circuit JSON init must be a list but was {init!r}.')
    init_ops = []
    for i in range(len(init)):
        state = init[i]
        q = devices.LineQubit(i)
        if state == 0:
            pass
        elif state == 1:
            init_ops.append(ops.X(q))
        elif state == '+':
            init_ops.append(ops.ry(np.pi / 2).on(q))
        elif state == '-':
            init_ops.append(ops.ry(-np.pi / 2).on(q))
        elif state == 'i':
            init_ops.append(ops.rx(-np.pi / 2).on(q))
        elif state == '-i':
            init_ops.append(ops.rx(np.pi / 2).on(q))
        else:
            raise ValueError(f'Unrecognized init state: {state!r}')
    return circuits.Moment(init_ops)