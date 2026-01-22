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
def _process_property_change(self, msg):
    msg = super()._process_property_change(msg)
    if self.object is not None:
        slice_params = {'slice_i': 0, 'slice_j': 1, 'slice_k': 2}
        for k, v in msg.items():
            sub_dim = self._subsample_dimensions
            ori_dim = self._orginal_dimensions
            if k in slice_params:
                index = slice_params[k]
                msg[k] = int(np.round(v * ori_dim[index] / sub_dim[index]))
    return msg