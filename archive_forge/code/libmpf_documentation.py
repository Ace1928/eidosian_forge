import math
from bisect import bisect
import sys
from .backend import (MPZ, MPZ_TYPE, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE,
from .libintmath import (giant_steps,
Compute the Euclidean norm sqrt(x**2 + y**2) of two raw mpfs
    x and y.