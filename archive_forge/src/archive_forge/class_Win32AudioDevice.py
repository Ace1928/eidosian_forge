from typing import List, Optional, Tuple
from pyglet.libs.win32 import com
from pyglet.libs.win32 import _ole32 as ole32
from pyglet.libs.win32.constants import CLSCTX_INPROC_SERVER
from pyglet.libs.win32.types import *
from pyglet.media.devices import base
from pyglet.util import debug_print
class Win32AudioDevice(base.AudioDevice):
    platform_state = {DEVICE_STATE_ACTIVE: base.DeviceState.ACTIVE, DEVICE_STATE_DISABLED: base.DeviceState.DISABLED, DEVICE_STATE_NOTPRESENT: base.DeviceState.MISSING, DEVICE_STATE_UNPLUGGED: base.DeviceState.UNPLUGGED}
    platform_flow = {eRender: base.DeviceFlow.OUTPUT, eCapture: base.DeviceFlow.INPUT, eAll: base.DeviceFlow.INPUT_OUTPUT}