import time
import os
import libevdev
from ._clib import Libevdev, UinputDevice
from ._clib import READ_FLAG_SYNC, READ_FLAG_NORMAL, READ_FLAG_FORCE_SYNC, READ_FLAG_BLOCKING
from .event import InputEvent
from .const import InputProperty
class _EventValueSet:

    def __init__(self, parent_device):
        self._device = parent_device

    def __getitem__(self, code):
        if code.type == libevdev.EV_ABS and code >= libevdev.EV_ABS.ABS_MT_SLOT and (self._device.num_slots is not None):
            raise InvalidArgumentException('Cannot fetch value for MT axes')
        return self._device._libevdev.event_value(code.type.value, code.value)