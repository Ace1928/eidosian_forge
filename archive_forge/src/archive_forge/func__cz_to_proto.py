import json
from typing import Any, cast, Dict, Optional, Sequence, Tuple, TYPE_CHECKING, Iterator
import numpy as np
import sympy
import cirq
from cirq_google.api.v1 import operations_pb2
def _cz_to_proto(gate: cirq.CZPowGate, p: cirq.Qid, q: cirq.Qid) -> operations_pb2.Exp11:
    return operations_pb2.Exp11(target1=_qubit_to_proto(p), target2=_qubit_to_proto(q), half_turns=_parameterized_value_to_proto(gate.exponent))