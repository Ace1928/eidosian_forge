from typing import List, Optional, Tuple
from pyglet.libs.win32 import com
from pyglet.libs.win32 import _ole32 as ole32
from pyglet.libs.win32.constants import CLSCTX_INPROC_SERVER
from pyglet.libs.win32.types import *
from pyglet.media.devices import base
from pyglet.util import debug_print
def OnDeviceRemoved(self, pwstrDeviceId):
    dev = self.audio_devices.remove_device(pwstrDeviceId)
    assert _debug(f'Audio device was removed {pwstrDeviceId} : {dev}')
    self.audio_devices.dispatch_event('on_device_removed', dev)