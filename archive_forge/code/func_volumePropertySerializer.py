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
def volumePropertySerializer(parent, propObj, propObjId, context, depth):
    dependencies = []
    calls = []
    ofun = propObj.GetScalarOpacity()
    if ofun:
        ofunId = context.getReferenceId(ofun)
        ofunInstance = serializeInstance(propObj, ofun, ofunId, context, depth + 1)
        if ofunInstance:
            dependencies.append(ofunInstance)
            calls.append(['setScalarOpacity', [0, wrapId(ofunId)]])
    ctfun = propObj.GetRGBTransferFunction()
    if ctfun:
        ctfunId = context.getReferenceId(ctfun)
        ctfunInstance = serializeInstance(propObj, ctfun, ctfunId, context, depth + 1)
        if ctfunInstance:
            dependencies.append(ctfunInstance)
            calls.append(['setRGBTransferFunction', [0, wrapId(ctfunId)]])
    calls += [['setScalarOpacityUnitDistance', [0, propObj.GetScalarOpacityUnitDistance(0)]], ['setComponentWeight', [0, propObj.GetComponentWeight(0)]], ['setUseGradientOpacity', [0, int(not propObj.GetDisableGradientOpacity())]]]
    return {'parent': context.getReferenceId(parent), 'id': propObjId, 'type': propObj.GetClassName(), 'properties': {'independentComponents': propObj.GetIndependentComponents(), 'interpolationType': propObj.GetInterpolationType(), 'ambient': propObj.GetAmbient(), 'diffuse': propObj.GetDiffuse(), 'shade': propObj.GetShade(), 'specular': propObj.GetSpecular(0), 'specularPower': propObj.GetSpecularPower()}, 'dependencies': dependencies, 'calls': calls}