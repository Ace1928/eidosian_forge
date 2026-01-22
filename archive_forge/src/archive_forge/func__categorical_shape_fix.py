import numpy as np
import six
from patsy import PatsyError
from patsy.util import (SortAnythingKey,
def _categorical_shape_fix(data):
    if hasattr(data, 'ndim') and data.ndim > 1:
        raise PatsyError('categorical data cannot be >1-dimensional')
    if not iterable(data) or isinstance(data, (six.text_type, six.binary_type)):
        data = [data]
    return data