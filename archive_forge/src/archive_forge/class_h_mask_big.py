import math
from bisect import bisect
import sys
from .backend import (MPZ, MPZ_TYPE, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE,
from .libintmath import (giant_steps,
class h_mask_big:

    def __getitem__(self, n):
        return (MPZ_ONE << n - 1) - 1