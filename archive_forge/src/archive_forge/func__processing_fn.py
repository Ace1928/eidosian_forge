from typing import Sequence, Callable
import functools
from functools import partial
from warnings import warn
import numpy as np
from scipy.special import factorial
from scipy.linalg import solve as linalg_solve
import pennylane as qml
from pennylane.measurements import ProbabilityMP
from pennylane import transform
from pennylane.transforms.tape_expand import expand_invalid_trainable
from pennylane.gradients.gradient_transform import _contract_qjac_with_cjac
from .general_shift_rules import generate_shifted_tapes
from .gradient_transform import (
def _processing_fn(results, shots, single_shot_batch_fn):
    if not shots.has_partitioned_shots:
        return single_shot_batch_fn(results)
    grads_tuple = []
    for idx in range(shots.num_copies):
        res = [tape_res[idx] for tape_res in results]
        g_tuple = single_shot_batch_fn(res)
        grads_tuple.append(g_tuple)
    return tuple(grads_tuple)