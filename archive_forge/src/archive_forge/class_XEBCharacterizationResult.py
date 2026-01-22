import dataclasses
from abc import abstractmethod, ABC
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import pandas as pd
import sympy
from cirq import circuits, ops, protocols, _import
from cirq.experiments.xeb_simulation import simulate_2q_xeb_circuits
@dataclasses.dataclass(frozen=True)
class XEBCharacterizationResult:
    """The result of `characterize_phased_fsim_parameters_with_xeb`.

    Attributes:
        optimization_results: A mapping from qubit pair to the raw scipy OptimizeResult object
        final_params: A mapping from qubit pair to a dictionary of (angle_name, angle_value)
            key-value pairs
        fidelities_df: A dataframe containing per-cycle_depth and per-pair fidelities after
            fitting the characterization.
    """
    optimization_results: Dict[QPair_T, 'scipy.optimize.OptimizeResult']
    final_params: Dict[QPair_T, Dict[str, float]]
    fidelities_df: pd.DataFrame