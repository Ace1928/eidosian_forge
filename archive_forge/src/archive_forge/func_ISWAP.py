from deprecated import deprecated
from deprecated.sphinx import versionadded
from numbers import Real
from typing import Callable, Mapping, Optional, Tuple, Union, Iterable, no_type_check
import numpy as np
from pyquil.quilatom import (
from pyquil.quilbase import (
def ISWAP(q1: QubitDesignator, q2: QubitDesignator) -> Gate:
    """Produces an ISWAP gate::

        ISWAP = [[1, 0,  0,  0],
                 [0, 0,  1j, 0],
                 [0, 1j, 0,  0],
                 [0, 0,  0,  1]]

    This gate swaps the state of two qubits, applying a -i phase to q1 when it
    is in the 1 state and a -i phase to q2 when it is in the 0 state.

    :param q1: Qubit 1.
    :param q2: Qubit 2.
    :returns: A Gate object.
    """
    return Gate(name='ISWAP', params=[], qubits=[unpack_qubit(q) for q in (q1, q2)])