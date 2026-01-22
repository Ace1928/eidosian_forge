import ctypes
import platform
import os
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.libs.win32 import com
from pyglet.util import debug_print
class X3DAUDIO_DSP_SETTINGS(Structure):
    _fields_ = [('pMatrixCoefficients', POINTER(FLOAT)), ('pDelayTimes', POINTER(FLOAT32)), ('SrcChannelCount', UINT32), ('DstChannelCount', UINT32), ('LPFDirectCoefficient', FLOAT32), ('LPFReverbCoefficient', FLOAT32), ('ReverbLevel', FLOAT32), ('DopplerFactor', FLOAT32), ('EmitterToListenerAngle', FLOAT32), ('EmitterToListenerDistance', FLOAT32), ('EmitterVelocityComponent', FLOAT32), ('ListenerVelocityComponent', FLOAT32)]