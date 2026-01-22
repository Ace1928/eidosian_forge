from typing import Dict, Optional
import pyglet
from pyglet.input import base
from pyglet.input.win32.directinput import _di_manager as _di_device_manager
from pyglet.input.win32.directinput import DirectInputDevice, _create_controller
from pyglet.input.win32.directinput import get_devices as dinput_get_devices
from pyglet.input.win32.directinput import get_controllers as dinput_get_controllers
from pyglet.input.win32.directinput import get_joysticks
def _add_di_controller(self, device: DirectInputDevice) -> Optional[base.Controller]:
    controller = _create_controller(device)
    if controller:
        self._di_controllers[device] = controller
        return controller
    return None