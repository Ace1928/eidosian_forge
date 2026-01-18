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
def zipCompression(name, data):
    with io.BytesIO() as in_memory:
        with zipfile.ZipFile(in_memory, mode='w') as zf:
            zf.writestr('data/%s' % name, data, zipfile.ZIP_DEFLATED)
        in_memory.seek(0)
        return in_memory.read()