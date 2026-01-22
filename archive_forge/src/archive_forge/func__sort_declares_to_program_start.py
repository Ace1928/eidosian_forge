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
def _sort_declares_to_program_start(self) -> None:
    """
        Re-order DECLARE instructions within this program to the beginning, followed by
        all other instructions. Reordering is stable among DECLARE and non-DECLARE instructions.
        """
    self._instructions = sorted(self._instructions, key=lambda instruction: not isinstance(instruction, Declare))