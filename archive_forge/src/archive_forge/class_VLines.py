from numbers import Number
import numpy as np
import param
from ..core import Dimension, Element, Element2D
from ..core.data import Dataset
from ..core.util import datetime_types
class VLines(VectorizedAnnotation):
    kdims = param.List(default=[Dimension('x')], bounds=(1, 1))
    group = param.String(default='VLines', constant=True)