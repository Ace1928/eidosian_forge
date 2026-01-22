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
def _expand_transform_finite_diff(tape: qml.tape.QuantumTape, argnum=None, h=1e-07, approx_order=1, n=1, strategy='forward', f0=None, validate_params=True) -> (Sequence[qml.tape.QuantumTape], Callable):
    """Expand function to be applied before finite difference."""
    expanded_tape = expand_invalid_trainable(tape)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]
    return ([expanded_tape], null_postprocessing)