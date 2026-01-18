import abc
import itertools
from typing import Iterable, Optional, TYPE_CHECKING, Tuple, cast
from cirq import devices, ops, value
from cirq.contrib.graph_device.hypergraph import UndirectedHypergraph
def validate_moment(self, moment: 'cirq.Moment'):
    super().validate_moment(moment)
    ops = moment.operations
    for i, op in enumerate(ops):
        other_ops = ops[:i] + ops[i + 1:]
        self.validate_crosstalk(op, other_ops)