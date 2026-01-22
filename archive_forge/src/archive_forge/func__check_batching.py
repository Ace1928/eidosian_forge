import abc
from typing import Callable, List
import copy
import pennylane as qml
from pennylane import math
from pennylane.operation import Operator, _UNSET_BATCH_SIZE
from pennylane.wires import Wires
def _check_batching(self):
    batch_sizes = {op.batch_size for op in self if op.batch_size is not None}
    if len(batch_sizes) > 1:
        raise ValueError(f'Broadcasting was attempted but the broadcasted dimensions do not match: {batch_sizes}.')
    self._batch_size = batch_sizes.pop() if batch_sizes else None