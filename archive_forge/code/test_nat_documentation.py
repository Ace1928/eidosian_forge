from datetime import (
import operator
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import iNaT
from pandas.compat.numpy import np_version_gte1p24p3
from pandas import (
import pandas._testing as tm
from pandas.core import roperator
from pandas.core.arrays import (

    Get overlapping public methods between NaT and another class.

    Parameters
    ----------
    klass : type
        The class to compare with NaT
    as_tuple : bool, default False
        Whether to return a list of tuples of the form (klass, method).

    Returns
    -------
    overlap : list
    