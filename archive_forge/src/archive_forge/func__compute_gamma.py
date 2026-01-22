from typing import Optional, List, Tuple, Iterable, Callable, Union, Dict
import numpy as np
from qiskit.exceptions import QiskitError
from ..distributions.quasi import QuasiDistribution
from ..counts import Counts
from .base_readout_mitigator import BaseReadoutMitigator
from .utils import counts_probability_vector, z_diagonal, str2diag
def _compute_gamma(self, qubits=None):
    """Compute gamma for N-qubit mitigation"""
    if qubits is None:
        gammas = self._gammas
    else:
        qubit_indices = [self._qubit_index[qubit] for qubit in qubits]
        gammas = self._gammas[qubit_indices]
    return np.prod(gammas)