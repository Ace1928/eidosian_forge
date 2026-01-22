from typing import List, Union, Tuple
import numpy as np
import pennylane as qml
from pennylane.ops import Sum, Hamiltonian, SProd, Prod
from pennylane.measurements import (
from pennylane.typing import TensorLike
from .apply_operation import apply_operation
from .measure import flatten_state
def _measure_hamiltonian_with_samples(mp: List[SampleMeasurement], state: np.ndarray, shots: Shots, is_state_batched: bool=False, rng=None, prng_key=None):
    mp = mp[0]

    def _sum_for_single_shot(s):
        results = measure_with_samples([ExpectationMP(t) for t in mp.obs.terms()[1]], state, s, is_state_batched=is_state_batched, rng=rng, prng_key=prng_key)
        return sum((c * res for c, res in zip(mp.obs.terms()[0], results)))
    unsqueezed_results = tuple((_sum_for_single_shot(type(shots)(s)) for s in shots))
    return [unsqueezed_results] if shots.has_partitioned_shots else [unsqueezed_results[0]]