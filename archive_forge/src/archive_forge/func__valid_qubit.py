import re
from itertools import product
import numpy as np
import copy
from typing import (
from pyquil.quilatom import (
from .quil import Program
from .gates import H, RZ, RX, CNOT, X, PHASE, QUANTUM_GATES
from numbers import Number, Complex
from collections import OrderedDict
import warnings
def _valid_qubit(index: Optional[Union[PauliTargetDesignator, QubitPlaceholder]]) -> bool:
    return isinstance(index, integer_types) and index >= 0 or isinstance(index, QubitPlaceholder) or isinstance(index, FormalArgument)