from statsmodels.compat.pandas import (
from abc import ABC, abstractmethod
import datetime as dt
from typing import Optional, Union
from collections.abc import Hashable, Sequence
import numpy as np
import pandas as pd
from scipy.linalg import qr
from statsmodels.iolib.summary import d_or_f
from statsmodels.tools.validation import (
from statsmodels.tsa.tsatools import freq_to_period
def _remove_zeros_ones(self, terms: pd.DataFrame) -> pd.DataFrame:
    all_zero = np.all(terms == 0, axis=0)
    if np.any(all_zero):
        terms = terms.loc[:, ~all_zero]
    is_constant = terms.max(axis=0) == terms.min(axis=0)
    if np.sum(is_constant) > 1:
        surplus_consts = is_constant & is_constant.duplicated()
        terms = terms.loc[:, ~surplus_consts]
    return terms