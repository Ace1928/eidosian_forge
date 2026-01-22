import dataclasses
from abc import abstractmethod, ABC
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import pandas as pd
import sympy
from cirq import circuits, ops, protocols, _import
from cirq.experiments.xeb_simulation import simulate_2q_xeb_circuits
class XEBCharacterizationOptions(ABC):

    @staticmethod
    @abstractmethod
    def should_parameterize(op: 'cirq.Operation') -> bool:
        """Whether to replace `op` with a parameterized version."""

    @abstractmethod
    def get_parameterized_gate(self) -> 'cirq.Gate':
        """The parameterized gate to use."""

    @abstractmethod
    def get_initial_simplex_and_names(self, initial_simplex_step_size: float=0.1) -> Tuple[np.ndarray, List[str]]:
        """Return an initial Nelder-Mead simplex and the names for each parameter."""