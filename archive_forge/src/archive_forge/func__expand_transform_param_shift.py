from typing import Sequence, Callable
from functools import partial
import numpy as np
import pennylane as qml
from pennylane.measurements import VarianceMP
from pennylane import transform
from pennylane.transforms.tape_expand import expand_invalid_trainable
from pennylane.gradients.gradient_transform import _contract_qjac_with_cjac
from .finite_difference import finite_diff
from .general_shift_rules import (
from .gradient_transform import (
def _expand_transform_param_shift(tape: qml.tape.QuantumTape, argnum=None, shifts=None, gradient_recipes=None, fallback_fn=finite_diff, f0=None, broadcast=False) -> (Sequence[qml.tape.QuantumTape], Callable):
    """Expand function to be applied before parameter shift."""
    expanded_tape = expand_invalid_trainable(tape)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]
    return ([expanded_tape], null_postprocessing)