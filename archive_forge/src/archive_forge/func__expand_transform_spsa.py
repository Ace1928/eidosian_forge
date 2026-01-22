from typing import Sequence, Callable
from functools import partial
import numpy as np
import pennylane as qml
from pennylane import transform
from pennylane.gradients.gradient_transform import _contract_qjac_with_cjac
from pennylane.transforms.tape_expand import expand_invalid_trainable
from .finite_difference import _processing_fn, finite_diff_coeffs
from .gradient_transform import (
from .general_shift_rules import generate_multishifted_tapes
def _expand_transform_spsa(tape: qml.tape.QuantumTape, argnum=None, h=1e-05, approx_order=2, n=1, strategy='center', f0=None, validate_params=True, num_directions=1, sampler=_rademacher_sampler, sampler_rng=None) -> (Sequence[qml.tape.QuantumTape], Callable):
    """Expand function to be applied before spsa gradient."""
    expanded_tape = expand_invalid_trainable(tape)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]
    return ([expanded_tape], null_postprocessing)