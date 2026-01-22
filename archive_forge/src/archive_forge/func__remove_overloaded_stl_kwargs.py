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
def _remove_overloaded_stl_kwargs(stl_kwargs: dict) -> dict:
    args = ['endog', 'period', 'seasonal']
    for arg in args:
        stl_kwargs.pop(arg, None)
    return stl_kwargs