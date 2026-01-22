from __future__ import annotations
from collections import abc
import numbers
import operator
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
import pandas._libs.sparse as splib
from pandas._libs.sparse import (
from pandas._libs.tslibs import NaT
from pandas.compat.numpy import function as nv
from pandas.errors import PerformanceWarning
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import (
from pandas.core.dtypes.astype import astype_array
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.core import arraylike
import pandas.core.algorithms as algos
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays import ExtensionArray
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.construction import (
from pandas.core.indexers import (
from pandas.core.nanops import check_below_min_count
from pandas.io.formats import printing
@property
def _valid_sp_values(self) -> np.ndarray:
    sp_vals = self.sp_values
    mask = notna(sp_vals)
    return sp_vals[mask]