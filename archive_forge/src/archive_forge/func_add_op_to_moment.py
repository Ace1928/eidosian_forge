from collections import defaultdict
import bisect
import dataclasses
from typing import (
from cirq import circuits, ops, protocols
from cirq.circuits.circuit import CIRCUIT_TYPE
def add_op_to_moment(self, moment_index: int, op: 'cirq.Operation') -> None:
    self.ops_by_index[moment_index][op] = 0
    for q in op.qubits:
        if moment_index > self.qubit_indexes[q][-1]:
            self.qubit_indexes[q].append(moment_index)
        else:
            bisect.insort(self.qubit_indexes[q], moment_index)
    for mkey in protocols.measurement_key_objs(op):
        bisect.insort(self.mkey_indexes[mkey], moment_index)
    for ckey in protocols.control_keys(op):
        bisect.insort(self.ckey_indexes[ckey], moment_index)