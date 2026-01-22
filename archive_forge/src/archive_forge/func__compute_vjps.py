import abc
import inspect
import logging
from typing import Tuple, Callable, Optional, Union
from cachetools import LRUCache
import numpy as np
import pennylane as qml
from pennylane.tape import QuantumScript
from pennylane.typing import ResultBatch, TensorLike
def _compute_vjps(jacs, dys, tapes):
    """Compute the vjps of multiple tapes, directly for a Jacobian and co-tangents dys."""
    f = {True: qml.gradients.compute_vjp_multi, False: qml.gradients.compute_vjp_single}
    vjps = []
    for jac, dy, t in zip(jacs, dys, tapes):
        multi = len(t.measurements) > 1
        if t.shots.has_partitioned_shots:
            shot_vjps = [f[multi](d, j) for d, j in zip(dy, jac)]
            vjps.append(qml.math.sum(qml.math.stack(shot_vjps), axis=0))
        else:
            vjps.append(f[multi](dy, jac))
    return tuple(vjps)