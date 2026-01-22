from typing import List, Optional, Tuple
from pyglet.libs.win32 import com
from pyglet.libs.win32 import _ole32 as ole32
from pyglet.libs.win32.constants import CLSCTX_INPROC_SERVER
from pyglet.libs.win32.types import *
from pyglet.media.devices import base
from pyglet.util import debug_print
def OnDefaultDeviceChanged(self, flow, role, pwstrDeviceId):
    if role == 0:
        if pwstrDeviceId is None:
            device = None
        else:
            device = self.audio_devices.get_cached_device(pwstrDeviceId)
        pyglet_flow = Win32AudioDevice.platform_flow[flow]
        assert _debug(f'Default device was changed to: {device} ({pyglet_flow})')
        self.audio_devices.dispatch_event('on_default_changed', device, pyglet_flow)