import time
import weakref
import threading
import pyglet
from pyglet.libs.win32 import com
from pyglet.event import EventDispatcher
from pyglet.libs.win32.types import *
from pyglet.libs.win32 import _ole32 as ole32, _oleaut32 as oleaut32
from pyglet.libs.win32.constants import CLSCTX_INPROC_SERVER
from pyglet.input.base import Device, Controller, Button, AbsoluteAxis, ControllerManager
class XInputDeviceManager(EventDispatcher):

    def __init__(self):
        self.all_devices = [XInputDevice(i, self) for i in range(XUSER_MAX_COUNT)]
        self._connected_devices = set()
        for i in range(XUSER_MAX_COUNT):
            device = self.all_devices[i]
            if XInputGetState(i, byref(device.xinput_state)) == ERROR_DEVICE_NOT_CONNECTED:
                continue
            device.connected = True
            self._connected_devices.add(i)
        self._polling_rate = 0.016
        self._detection_rate = 2.0
        self._exit = threading.Event()
        self._dev_lock = threading.Lock()
        self._thread = threading.Thread(target=self._get_state, daemon=True)
        self._thread.start()

    def get_devices(self):
        with self._dev_lock:
            return [dev for dev in self.all_devices if dev.connected]

    def _get_state(self):
        xuser_max_count = set(range(XUSER_MAX_COUNT))
        polling_rate = self._polling_rate
        detect_rate = self._detection_rate
        elapsed = 0.0
        while not self._exit.is_set():
            self._dev_lock.acquire()
            elapsed += polling_rate
            if elapsed >= detect_rate:
                for i in xuser_max_count - self._connected_devices:
                    device = self.all_devices[i]
                    if XInputGetState(i, byref(device.xinput_state)) == ERROR_DEVICE_NOT_CONNECTED:
                        continue
                    device.connected = True
                    self._connected_devices.add(i)
                    pyglet.app.platform_event_loop.post_event(self, 'on_connect', device)
                elapsed = 0.0
            for i in self._connected_devices.copy():
                device = self.all_devices[i]
                result = XInputGetState(i, byref(device.xinput_state))
                if result == ERROR_DEVICE_NOT_CONNECTED:
                    if device.connected:
                        device.connected = False
                        device.close()
                        self._connected_devices.remove(i)
                        pyglet.app.platform_event_loop.post_event(self, 'on_disconnect', device)
                        continue
                elif result == ERROR_SUCCESS and device.is_open:
                    if device.weak_duration:
                        device.weak_duration -= polling_rate
                        if device.weak_duration <= 0:
                            device.weak_duration = None
                            device.vibration.wRightMotorSpeed = 0
                            device.set_rumble_state()
                    if device.strong_duration:
                        device.strong_duration -= polling_rate
                        if device.strong_duration <= 0:
                            device.strong_duration = None
                            device.vibration.wLeftMotorSpeed = 0
                            device.set_rumble_state()
                    if device.xinput_state.dwPacketNumber == device.packet_number:
                        continue
                    pyglet.app.platform_event_loop.post_event(self, '_on_state_change', device)
            self._dev_lock.release()
            time.sleep(polling_rate)

    @staticmethod
    def _on_state_change(device):
        for button, name in controller_api_to_pyglet.items():
            device.controls[name].value = device.xinput_state.Gamepad.wButtons & button
        device.controls['lefttrigger'].value = device.xinput_state.Gamepad.bLeftTrigger
        device.controls['righttrigger'].value = device.xinput_state.Gamepad.bRightTrigger
        device.controls['leftx'].value = device.xinput_state.Gamepad.sThumbLX
        device.controls['lefty'].value = device.xinput_state.Gamepad.sThumbLY
        device.controls['rightx'].value = device.xinput_state.Gamepad.sThumbRX
        device.controls['righty'].value = device.xinput_state.Gamepad.sThumbRY
        device.packet_number = device.xinput_state.dwPacketNumber

    def on_connect(self, device):
        """A device was connected."""

    def on_disconnect(self, device):
        """A device was disconnected"""