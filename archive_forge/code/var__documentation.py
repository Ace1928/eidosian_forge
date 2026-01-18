from __future__ import annotations
from typing import TYPE_CHECKING
import numba
import numpy as np
from pandas.core._numba.kernels.shared import is_monotonic_increasing

Numba 1D var kernels that can be shared by
* Dataframe / Series
* groupby
* rolling / expanding

Mirrors pandas/_libs/window/aggregation.pyx
