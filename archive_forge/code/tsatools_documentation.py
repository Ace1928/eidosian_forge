from __future__ import annotations
from statsmodels.compat.python import lrange
import warnings
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
from typing import Literal
from statsmodels.tools.data import _is_recarray, _is_using_pandas
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tools.typing import NDArray
from statsmodels.tools.validation import (

    Convert a pandas frequency to a periodicity

    Parameters
    ----------
    freq : str or offset
        Frequency to convert

    Returns
    -------
    int
        Periodicity of freq

    Notes
    -----
    Annual maps to 1, quarterly maps to 4, monthly to 12, weekly to 52.
    