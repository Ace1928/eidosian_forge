import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def update_averages(self, ema: FloatsT, weights: FloatsT, t: int, max_decay: float=0.9999) -> None:
    decay = (1.0 + t) / (10.0 + t)
    if decay > max_decay:
        decay = max_decay
    ema -= (1 - decay) * (ema - weights)