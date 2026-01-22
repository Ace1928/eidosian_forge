from typing import cast, Dict, Hashable, Iterable, List, Optional, Sequence
from collections import OrderedDict
import dataclasses
import numpy as np
import cirq
from cirq_google.api import v2
from cirq_google.api.v2 import result_pb2
def _grid_qubits(op: cirq.Operation) -> List[cirq.GridQubit]:
    if not all((isinstance(q, cirq.GridQubit) for q in op.qubits)):
        raise ValueError(f'Expected GridQubits: {op.qubits}')
    return cast(List[cirq.GridQubit], list(op.qubits))