import numpy as np
from ..core import Dataset, Dimension, util
from ..element import Bars, Graph
from ..element.util import categorical_aggregate2d
from .util import get_axis_padding
class MultiDistributionMixin:

    def _get_axis_dims(self, element):
        return (element.kdims, element.vdims[0])

    def get_extents(self, element, ranges, range_type='combined', **kwargs):
        return super().get_extents(element, ranges, range_type, 'categorical', ydim=element.vdims[0])