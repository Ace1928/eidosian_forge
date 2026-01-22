from typing import Sequence, Callable
from itertools import chain
from functools import partial
import numpy as np
import pennylane as qml
from pennylane.gradients.metric_tensor import _contract_metric_tensor_with_cjac
from pennylane.transforms import transform
def _group_operations(tape):
    """Divide all operations of a tape into trainable operations and blocks
    of untrainable operations after each trainable one."""
    ops = tape.operations
    trainable_par_info = [tape._par_info[i] for i in tape.trainable_params]
    trainables = [info['op_idx'] for info in trainable_par_info]
    split_ids = list(chain.from_iterable(([idx, idx + 1] for idx in trainables)))
    all_groups = np.split(ops, split_ids)
    group_after_trainable_op = dict(enumerate(all_groups[::2], start=-1))
    trainable_operations = list(chain.from_iterable(all_groups[1::2]))
    return (trainable_operations, group_after_trainable_op)