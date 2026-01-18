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
@trainable_params.setter
def trainable_params(self, param_indices):
    """Store the indices of parameters that support differentiability.

        Args:
            param_indices (list[int]): parameter indices
        """
    if any((not isinstance(i, int) or i < 0 for i in param_indices)):
        raise ValueError('Argument indices must be non-negative integers.')
    num_params = len(self._par_info)
    if any((i > num_params for i in param_indices)):
        raise ValueError(f'Quantum Script only has {num_params} parameters.')
    self._trainable_params = sorted(set(param_indices))