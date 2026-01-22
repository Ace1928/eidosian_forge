from deprecated import deprecated
from deprecated.sphinx import versionadded
from numbers import Real
from typing import Callable, Mapping, Optional, Tuple, Union, Iterable, no_type_check
import numpy as np
from pyquil.quilatom import (
from pyquil.quilbase import (
def CSWAP(control: QubitDesignator, target_1: QubitDesignator, target_2: QubitDesignator) -> Gate:
    """Produces a controlled-SWAP gate. This gate conditionally swaps the state of two qubits::

        CSWAP = [[1, 0, 0, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 1]]


    :param control: The control qubit.
    :param target_1: The first target qubit.
    :param target_2: The second target qubit. The two target states are swapped if the control is
        in the ``|1>`` state.
    """
    qubits = [unpack_qubit(q) for q in (control, target_1, target_2)]
    return Gate(name='CSWAP', params=[], qubits=qubits)