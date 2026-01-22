import base64
import hashlib
import io
import struct
import time
import zipfile
from vtk.vtkCommonCore import vtkTypeInt32Array, vtkTypeUInt32Array
from vtk.vtkCommonDataModel import vtkDataObject
from vtk.vtkFiltersGeometry import (
from vtk.vtkRenderingCore import vtkColorTransferFunction
from .enums import TextPosition
def annotationSerializer(parent, prop, propId, context, depth):
    if context.debugSerializers:
        print('%s!!!Annotations are not handled directly by vtk.js but by bokeh model' % pad(depth))
    context.addAnnotation(parent, prop, propId)
    return None