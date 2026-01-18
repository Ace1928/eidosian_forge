import itertools
import types
import warnings
from collections import defaultdict
from typing import (
import numpy as np
from rpcq.messages import NativeQuilMetadata, ParameterAref
from pyquil._parser.parser import run_parser
from pyquil._memory import Memory
from pyquil.gates import MEASURE, RESET, MOVE
from pyquil.noise import _check_kraus_ops, _create_kraus_pragmas, pauli_kraus_map
from pyquil.quilatom import (
from pyquil.quilbase import (
from pyquil.quiltcalibrations import (
def wrap_in_numshots_loop(self, shots: int) -> 'Program':
    """
        Wraps a Quil program in a loop that re-runs the same program many times.

        Note: this function is a prototype of what will exist in the future when users will
        be responsible for writing this loop instead of having it happen automatically.

        :param shots: Number of iterations to loop through.
        """
    self.num_shots = shots
    return self