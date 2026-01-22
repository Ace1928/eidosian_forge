from numbers import Number
import numpy as np
import param
from ..core import Dimension, Element, Element2D
from ..core.data import Dataset
from ..core.util import datetime_types
class VSpans(VectorizedAnnotation):
    kdims = param.List(default=[Dimension('x0'), Dimension('x1')], bounds=(2, 2))
    group = param.String(default='VSpans', constant=True)