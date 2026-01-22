import numpy as np
import param
from ..core.data import Dataset
from ..core.dimension import Dimension, process_dimensions
from ..core.element import Element, Element2D
from ..core.util import get_param_values, unique_iterator
from .selection import Selection1DExpr, Selection2DExpr
class BoxWhisker(Selection1DExpr, Dataset, Element2D):
    """
    BoxWhisker represent data as a distributions highlighting the
    median, mean and various percentiles. It may have a single value
    dimension and any number of key dimensions declaring the grouping
    of each violin.
    """
    group = param.String(default='BoxWhisker', constant=True)
    kdims = param.List(default=[], bounds=(0, None))
    vdims = param.List(default=[Dimension('y')], bounds=(1, 1))
    _inverted_expr = True