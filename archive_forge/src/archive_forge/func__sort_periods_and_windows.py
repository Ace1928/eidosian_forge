from typing import Optional, Union
from collections.abc import Sequence
import warnings
import numpy as np
import pandas as pd
from scipy.stats import boxcox
from statsmodels.tools.typing import ArrayLike1D
from statsmodels.tsa.stl._stl import STL
from statsmodels.tsa.tsatools import freq_to_period
@staticmethod
def _sort_periods_and_windows(periods, windows) -> tuple[Sequence[int], Sequence[int]]:
    if len(periods) != len(windows):
        raise ValueError('Periods and windows must have same length')
    periods, windows = zip(*sorted(zip(periods, windows)))
    return (periods, windows)