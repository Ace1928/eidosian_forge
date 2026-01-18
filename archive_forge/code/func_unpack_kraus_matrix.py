import sys
from collections import namedtuple
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union, cast
import numpy as np
from pyquil.external.rpcq import CompilerISA
from pyquil.gates import I, RX, MEASURE
from pyquil.noise_gates import _get_qvm_noise_supported_gates
from pyquil.quilatom import MemoryReference, format_parameter, ParameterDesignator
from pyquil.quilbase import Pragma, Gate, Declare
@staticmethod
def unpack_kraus_matrix(m: Union[List[Any], np.ndarray]) -> np.ndarray:
    """
        Helper to optionally unpack a JSON compatible representation of a complex Kraus matrix.

        :param m: The representation of a Kraus operator. Either a complex
            square matrix (as numpy array or nested lists) or a JSON-able pair of real matrices
            (as nested lists) representing the element-wise real and imaginary part of m.
        :return: A complex square numpy array representing the Kraus operator.
        """
    m = np.asarray(m, dtype=complex)
    if m.ndim == 3:
        m = m[0] + 1j * m[1]
    if not m.ndim == 2:
        raise ValueError('Need 2d array.')
    if not m.shape[0] == m.shape[1]:
        raise ValueError('Need square matrix.')
    return m