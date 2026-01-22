import sys
from collections import namedtuple
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union, cast
import numpy as np
from pyquil.external.rpcq import CompilerISA
from pyquil.gates import I, RX, MEASURE
from pyquil.noise_gates import _get_qvm_noise_supported_gates
from pyquil.quilatom import MemoryReference, format_parameter, ParameterDesignator
from pyquil.quilbase import Pragma, Gate, Declare
def _create_kraus_pragmas(name: str, qubit_indices: Sequence[int], kraus_ops: Sequence[np.ndarray]) -> List[Pragma]:
    """
    Generate the pragmas to define a Kraus map for a specific gate on some qubits.

    :param name: The name of the gate.
    :param qubit_indices: The qubits
    :param kraus_ops: The Kraus operators as matrices.
    :return: A QUIL string with PRAGMA ADD-KRAUS ... statements.
    """
    pragmas = [Pragma('ADD-KRAUS', (name,) + tuple(qubit_indices), '({})'.format(' '.join(map(format_parameter, np.ravel(k))))) for k in kraus_ops]
    return pragmas