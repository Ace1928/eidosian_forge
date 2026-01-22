import itertools
import math
from functools import wraps
import numpy
import scipy.special as special
from .._config import get_config
from .fixes import parse_version
def _estimator_with_converted_arrays(estimator, converter):
    """Create new estimator which converting all attributes that are arrays.

    The converter is called on all NumPy arrays and arrays that support the
    `DLPack interface <https://dmlc.github.io/dlpack/latest/>`__.

    Parameters
    ----------
    estimator : Estimator
        Estimator to convert

    converter : callable
        Callable that takes an array attribute and returns the converted array.

    Returns
    -------
    new_estimator : Estimator
        Convert estimator
    """
    from sklearn.base import clone
    new_estimator = clone(estimator)
    for key, attribute in vars(estimator).items():
        if hasattr(attribute, '__dlpack__') or isinstance(attribute, numpy.ndarray):
            attribute = converter(attribute)
        setattr(new_estimator, key, attribute)
    return new_estimator