import dataclasses
from abc import abstractmethod, ABC
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import pandas as pd
import sympy
from cirq import circuits, ops, protocols, _import
from cirq.experiments.xeb_simulation import simulate_2q_xeb_circuits
@dataclasses.dataclass(frozen=True)
class _CharacterizePhasedFsimParametersWithXebClosure:
    """A closure object to wrap `characterize_phased_fsim_parameters_with_xeb` for use in
    multiprocessing."""
    parameterized_circuits: List['cirq.Circuit']
    cycle_depths: Sequence[int]
    options: XEBCharacterizationOptions
    initial_simplex_step_size: float = 0.1
    xatol: float = 0.001
    fatol: float = 0.001

    def __call__(self, sampled_df) -> XEBCharacterizationResult:
        return characterize_phased_fsim_parameters_with_xeb(sampled_df=sampled_df, parameterized_circuits=self.parameterized_circuits, cycle_depths=self.cycle_depths, options=self.options, initial_simplex_step_size=self.initial_simplex_step_size, xatol=self.xatol, fatol=self.fatol, verbose=False, pool=None)