import time
import os
import libevdev
from ._clib import Libevdev, UinputDevice
from ._clib import READ_FLAG_SYNC, READ_FLAG_NORMAL, READ_FLAG_FORCE_SYNC, READ_FLAG_BLOCKING
from .event import InputEvent
from .const import InputProperty
class DeviceGrabError(Exception):
    """
    A device grab failed to be issued. A caller must not assume that it has
    exclusive access to the events on the device.
    """