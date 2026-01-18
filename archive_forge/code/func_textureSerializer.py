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
def textureSerializer(parent, texture, textureId, context, depth):
    instance = genericMapperSerializer(parent, texture, textureId, context, depth)
    if not instance:
        return
    instance['type'] = texture.GetClassName()
    instance['properties'].update({'interpolate': texture.GetInterpolate(), 'repeat': texture.GetRepeat(), 'edgeClamp': texture.GetEdgeClamp()})
    return instance