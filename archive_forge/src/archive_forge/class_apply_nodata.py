import bisect
import re
import traceback
import warnings
from collections import defaultdict, namedtuple
import numpy as np
import param
from packaging.version import Version
from ..core import (
from ..core.ndmapping import item_check
from ..core.operation import Operation
from ..core.options import CallbackError, Cycle
from ..core.spaces import get_nested_streams
from ..core.util import (
from ..element import Points
from ..streams import LinkedStream, Params
from ..util.transform import dim
class apply_nodata(Operation):
    link_inputs = param.Boolean(default=True)
    nodata = param.Integer(default=None, doc='\n        Optional missing-data value for integer data.\n        If non-None, data with this value will be replaced with NaN so\n        that it is transparent (by default) when plotted.')

    def _replace_value(self, data):
        """Replace `nodata` value in data with NaN, if specified in opts"""
        data = data.astype('float64')
        mask = data != self.p.nodata
        if hasattr(data, 'where'):
            return data.where(mask, np.nan)
        return np.where(mask, data, np.nan)

    def _process(self, element, key=None):
        if self.p.nodata is None:
            return element
        if hasattr(element, 'interface'):
            vdim = element.vdims[0]
            dtype = element.interface.dtype(element, vdim)
            if dtype.kind not in 'iu':
                return element
            transform = dim(vdim, self._replace_value)
            return element.transform(**{vdim.name: transform})
        else:
            array = element.dimension_values(2, flat=False).T
            if array.dtype.kind not in 'iu':
                return element
            array = array.astype('float64')
            return element.clone(self._replace_value(array))