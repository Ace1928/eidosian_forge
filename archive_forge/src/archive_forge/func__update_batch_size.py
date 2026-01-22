import contextlib
import copy
from collections import Counter
from typing import List, Union, Optional, Sequence
import pennylane as qml
from pennylane.measurements import (
from pennylane.typing import TensorLike
from pennylane.operation import Observable, Operator, Operation, _UNSET_BATCH_SIZE
from pennylane.pytrees import register_pytree
from pennylane.queuing import AnnotatedQueue, process_queue
from pennylane.wires import Wires
def _update_batch_size(self):
    """Infer the batch_size of the quantum script from the batch sizes of its operations
        and check the latter for consistency.

        Sets:
            _batch_size (int): The common batch size of the quantum script operations, if any has one
        """
    candidate = None
    for op in self.operations:
        op_batch_size = getattr(op, 'batch_size', None)
        if op_batch_size is None:
            continue
        if candidate:
            if op_batch_size != candidate:
                raise ValueError(f'The batch sizes of the quantum script operations do not match, they include {candidate} and {op_batch_size}.')
        else:
            candidate = op_batch_size
    self._batch_size = candidate