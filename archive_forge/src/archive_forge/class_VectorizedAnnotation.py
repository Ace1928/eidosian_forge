from numbers import Number
import numpy as np
import param
from ..core import Dimension, Element, Element2D
from ..core.data import Dataset
from ..core.util import datetime_types
class VectorizedAnnotation(Dataset, Element2D):
    _auto_indexable_1d = False