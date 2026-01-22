import re
from fractions import Fraction
from typing import (
import numpy as np
import sympy
from typing_extensions import Protocol
from cirq import protocols, value
from cirq._doc import doc_private
def _op_info_with_fallback(op: 'cirq.Operation', args: 'cirq.CircuitDiagramInfoArgs') -> 'cirq.CircuitDiagramInfo':
    info = protocols.circuit_diagram_info(op, args, None)
    rows: List[LabelEntity] = list(op.qubits)
    if args.label_map is not None:
        rows += protocols.measurement_keys_touched(op) & args.label_map.keys()
    if info is not None:
        if max(1, len(rows)) != len(info.wire_symbols):
            raise ValueError(f'Wanted diagram info from {op!r} for {rows!r}) but got {info!r}')
        return info
    name = str(op.untagged)
    redundant_tail = f'({', '.join((str(e) for e in op.qubits))})'
    if name.endswith(redundant_tail):
        name = name[:-len(redundant_tail)]
    if op.tags:
        name += f'{list(op.tags)}'
    symbols = (name,) + tuple((f'#{i + 1}' for i in range(1, len(op.qubits))))
    return protocols.CircuitDiagramInfo(wire_symbols=symbols)