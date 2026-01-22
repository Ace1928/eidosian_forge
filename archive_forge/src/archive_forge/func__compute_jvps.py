import abc
import inspect
import logging
from typing import Tuple, Callable, Optional, Union
from cachetools import LRUCache
import numpy as np
import pennylane as qml
from pennylane.tape import QuantumScript
from pennylane.typing import ResultBatch, TensorLike
def _compute_jvps(jacs, tangents, tapes):
    """Compute the jvps of multiple tapes, directly for a Jacobian and tangents."""
    f = {True: qml.gradients.compute_jvp_multi, False: qml.gradients.compute_jvp_single}
    jvps = []
    for jac, dx, t in zip(jacs, tangents, tapes):
        multi = len(t.measurements) > 1
        if len(t.trainable_params) == 0:
            empty_shots = qml.measurements.Shots(None)
            zeros_jvp = tuple((np.zeros(mp.shape(None, empty_shots), dtype=mp.numeric_type) for mp in t.measurements))
            zeros_jvp = zeros_jvp[0] if len(t.measurements) == 1 else zeros_jvp
            if t.shots.has_partitioned_shots:
                jvps.append(tuple((zeros_jvp for _ in range(t.shots.num_copies))))
            else:
                jvps.append(zeros_jvp)
        elif t.shots.has_partitioned_shots:
            jvps.append(tuple((f[multi](dx, j) for j in jac)))
        else:
            jvps.append(f[multi](dx, jac))
    return tuple(jvps)