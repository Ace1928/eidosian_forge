import libevdev
import os
import ctypes
import errno
from ctypes import c_char_p
from ctypes import c_int
from ctypes import c_uint
from ctypes import c_void_p
from ctypes import c_long
from ctypes import c_int32
from ctypes import c_uint16
@classmethod
def type_max(cls, type):
    """
        :param type: the EV_<*> event type
        :return: the maximum code for this type or ``None`` if the type is
                 invalid
        """
    if not isinstance(type, int):
        type = cls.event_to_value(type)
    m = cls._event_type_get_max(type)
    return m if m > -1 else None