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
def _volume_from_array(self, sub_array):
    return dict(buffer=base64encode(sub_array.ravel(order='F')), dims=sub_array.shape, spacing=self._sub_spacing, origin=self.origin, data_range=(np.nanmin(sub_array), np.nanmax(sub_array)), dtype=sub_array.dtype.name)