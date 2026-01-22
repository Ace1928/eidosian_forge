import collections
import json
from numbers import Complex
from typing import (
import numpy as np
from deprecated import deprecated
from deprecated.sphinx import versionadded
from pyquil.quilatom import (
from dataclasses import dataclass
@deprecated(version='3.5.1', reason='The correct instruction is SWAP-PHASES, not SWAP-PHASE')
class SwapPhase(SwapPhases):
    pass