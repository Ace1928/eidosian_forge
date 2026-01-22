from deprecated import deprecated
from deprecated.sphinx import versionadded
from numbers import Real
from typing import Callable, Mapping, Optional, Tuple, Union, Iterable, no_type_check
import numpy as np
from pyquil.quilatom import (
from pyquil.quilbase import (
@versionadded(version='3.5.1', reason='The correct instruction is SWAP-PHASES, not SWAP-PHASE')
def SWAP_PHASES(frameA: Frame, frameB: Frame) -> SwapPhases:
    """
    Produce a SWAP-PHASES instruction.

    :param frameA: A frame.
    :param frameB: A frame.
    :returns: A SwapPhases instance.
    """
    return SwapPhases(frameA, frameB)