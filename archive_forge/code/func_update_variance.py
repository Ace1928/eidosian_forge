import math
from collections import OrderedDict
import numpy
from .base import numeric_types, string_types
from . import ndarray
from . import registry
def update_variance(self, new_values, *aggregate):
    count, mean, m_2 = aggregate
    count += len(new_values)
    delta = new_values - mean
    mean += numpy.sum(delta / count)
    delta_2 = new_values - mean
    m_2 += numpy.sum(delta * delta_2)
    return (count, mean, m_2)