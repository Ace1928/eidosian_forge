from typing import List, Optional, Tuple
from pyglet.libs.win32 import com
from pyglet.libs.win32 import _ole32 as ole32
from pyglet.libs.win32.constants import CLSCTX_INPROC_SERVER
from pyglet.libs.win32.types import *
from pyglet.media.devices import base
from pyglet.util import debug_print
class AudioNotificationCB(com.COMObject):
    _interfaces_ = [IMMNotificationClient]

    def __init__(self, audio_devices: 'Win32AudioDeviceManager'):
        super().__init__()
        self.audio_devices = audio_devices
        self._lost = False

    def OnDeviceStateChanged(self, pwstrDeviceId, dwNewState):
        device = self.audio_devices.get_cached_device(pwstrDeviceId)
        old_state = device.state
        pyglet_old_state = Win32AudioDevice.platform_state[old_state]
        pyglet_new_state = Win32AudioDevice.platform_state[dwNewState]
        assert _debug(f"Audio device '{device.name}' changed state. From: {pyglet_old_state} to: {pyglet_new_state}")
        device.state = dwNewState
        self.audio_devices.dispatch_event('on_device_state_changed', device, pyglet_old_state, pyglet_new_state)

    def OnDeviceAdded(self, pwstrDeviceId):
        dev = self.audio_devices.add_device(pwstrDeviceId)
        assert _debug(f'Audio device was added {pwstrDeviceId}: {dev}')
        self.audio_devices.dispatch_event('on_device_added', dev)

    def OnDeviceRemoved(self, pwstrDeviceId):
        dev = self.audio_devices.remove_device(pwstrDeviceId)
        assert _debug(f'Audio device was removed {pwstrDeviceId} : {dev}')
        self.audio_devices.dispatch_event('on_device_removed', dev)

    def OnDefaultDeviceChanged(self, flow, role, pwstrDeviceId):
        if role == 0:
            if pwstrDeviceId is None:
                device = None
            else:
                device = self.audio_devices.get_cached_device(pwstrDeviceId)
            pyglet_flow = Win32AudioDevice.platform_flow[flow]
            assert _debug(f'Default device was changed to: {device} ({pyglet_flow})')
            self.audio_devices.dispatch_event('on_default_changed', device, pyglet_flow)

    def OnPropertyValueChanged(self, pwstrDeviceId, key):
        pass