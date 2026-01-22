from six.moves import queue
from six.moves import range
import ctypes
import ctypes.util
import logging
import sys
import threading
from pyu2f import errors
from pyu2f.hid import base
def GetDevicePath(device_handle):
    """Obtains the unique path for the device.

  Args:
    device_handle: reference to the device

  Returns:
    A unique path for the device, obtained from the IO Registry

  """
    io_service_obj = iokit.IOHIDDeviceGetService(device_handle)
    str_buffer = ctypes.create_string_buffer(DEVICE_PATH_BUFFER_SIZE)
    iokit.IORegistryEntryGetPath(io_service_obj, K_IO_SERVICE_PLANE, str_buffer)
    return str_buffer.value