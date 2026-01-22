from __future__ import annotations
import base64
import json
import sys
import zipfile
from abc import abstractmethod
from typing import (
from urllib.request import urlopen
import numpy as np
import param
from bokeh.models import LinearColorMapper
from bokeh.util.serialization import make_globally_unique_id
from pyviz_comms import JupyterComm
from ...param import ParamMethod
from ...util import isfile, lazy_load
from ..base import PaneBase
from ..plot import Bokeh
from .enums import PRESET_CMAPS
def _subsample_array(self, array):
    original_shape = array.shape
    spacing = self.spacing
    extent = tuple(((o_s - 1) * s for o_s, s in zip(original_shape, spacing)))
    dim_ratio = np.cbrt(array.nbytes / 1000000.0 / self.max_data_size)
    max_shape = tuple((int(o_s / dim_ratio) for o_s in original_shape))
    dowsnscale_factor = [max(o_s, m_s) / m_s for m_s, o_s in zip(max_shape, original_shape)]
    if any([d_f > 1 for d_f in dowsnscale_factor]):
        try:
            import scipy.ndimage as nd
            sub_array = nd.interpolation.zoom(array, zoom=[1 / d_f for d_f in dowsnscale_factor], order=0, mode='nearest')
        except ImportError:
            sub_array = array[::int(np.ceil(dowsnscale_factor[0])), ::int(np.ceil(dowsnscale_factor[1])), ::int(np.ceil(dowsnscale_factor[2]))]
        self._sub_spacing = tuple((e / (s - 1) for e, s in zip(extent, sub_array.shape)))
    else:
        sub_array = array
        self._sub_spacing = self.spacing
    return sub_array