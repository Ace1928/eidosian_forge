from typing import Optional, Union
from collections.abc import Sequence
import warnings
import numpy as np
import pandas as pd
from scipy.stats import boxcox
from statsmodels.tools.typing import ArrayLike1D
from statsmodels.tsa.stl._stl import STL
from statsmodels.tsa.tsatools import freq_to_period

        Estimate a trend component, multiple seasonal components, and a
        residual component.

        Returns
        -------
        DecomposeResult
            Estimation results.
        