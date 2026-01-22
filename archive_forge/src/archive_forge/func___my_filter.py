from decorator import decorator
import numpy as np
import scipy
import scipy.signal
import scipy.ndimage
import sklearn
import sklearn.cluster
import sklearn.feature_extraction
import sklearn.neighbors
from ._cache import cache
from . import util
from .filters import diagonal_filter
from .util.exceptions import ParameterError
from typing import Any, Callable, Optional, TypeVar, Union, overload
from typing_extensions import Literal
from ._typing import _WindowSpec, _FloatLike_co
def __my_filter(wrapped_f, *args, **kwargs):
    """Wrap the filter with lag conversions"""
    args = list(args)
    args[index] = recurrence_to_lag(args[index], pad=pad)
    result = wrapped_f(*args, **kwargs)
    return lag_to_recurrence(result)