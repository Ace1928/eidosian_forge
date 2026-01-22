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
def _get_vtkjs(self, fetch=True):
    data_path, data_url = (None, None)
    if isinstance(self.object, str) and self.object.endswith('.vtkjs'):
        data_path = data_path
        if not isfile(self.object):
            data_url = self.object
    if self._vtkjs is None and self.object is not None:
        vtkjs = None
        if data_url and fetch:
            vtkjs = urlopen(data_url).read() if fetch else data_url
        elif data_path:
            with open(self.object, 'rb') as f:
                vtkjs = f.read()
        elif hasattr(self.object, 'read'):
            vtkjs = self.object.read()
        self._vtkjs = vtkjs
    return (data_url, self._vtkjs)