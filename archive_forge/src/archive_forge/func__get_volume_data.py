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
def _get_volume_data(self):
    if self.object is None:
        return None
    elif isinstance(self.object, np.ndarray):
        return self._volume_from_array(self._subsample_array(self.object))
    else:
        available_serializer = [v for k, v in VTKVolume._serializers.items() if isinstance(self.object, k)]
        if not available_serializer:
            import vtk
            from vtk.util import numpy_support

            def volume_serializer(inst):
                imageData = inst.object
                array = numpy_support.vtk_to_numpy(imageData.GetPointData().GetScalars())
                dims = imageData.GetDimensions()
                inst.spacing = imageData.GetSpacing()
                inst.origin = imageData.GetOrigin()
                return inst._volume_from_array(inst._subsample_array(array.reshape(dims, order='F')))
            VTKVolume.register_serializer(vtk.vtkImageData, volume_serializer)
            serializer = volume_serializer
        else:
            serializer = available_serializer[0]
        return serializer(self)