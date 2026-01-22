from itertools import permutations
import numpy as np
import math
from ._continuous_distns import norm
import scipy.stats
from dataclasses import dataclass
@dataclass
class PageTrendTestResult:
    statistic: float
    pvalue: float
    method: str