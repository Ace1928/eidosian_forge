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
def _serialize_ren_win(self, ren_win, context, binary=False, compression=True, exclude_arrays=None):
    import panel.pane.vtk.synchronizable_serializer as rws
    if exclude_arrays is None:
        exclude_arrays = []
    ren_win.OffScreenRenderingOn()
    ren_win.Modified()
    ren_win.Render()
    scene = rws.serializeInstance(None, ren_win, context.getReferenceId(ren_win), context, 0)
    scene['properties']['numberOfLayers'] = 2
    arrays = {name: context.getCachedDataArray(name, binary=True, compression=False) for name in context.dataArrayCache.keys() if name not in exclude_arrays}
    annotations = context.getAnnotations()
    return (scene, arrays, annotations)