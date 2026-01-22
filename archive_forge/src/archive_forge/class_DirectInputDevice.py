import ctypes
import warnings
from typing import List, Dict, Optional
from pyglet.libs.win32.constants import WM_DEVICECHANGE, DBT_DEVICEARRIVAL, DBT_DEVICEREMOVECOMPLETE, \
from pyglet.event import EventDispatcher
import pyglet
from pyglet.input import base
from pyglet.libs import win32
from pyglet.libs.win32 import dinput, _user32, DEV_BROADCAST_DEVICEINTERFACE, com, DEV_BROADCAST_HDR
from pyglet.libs.win32 import _kernel32
from pyglet.input.controller import get_mapping
from pyglet.input.base import ControllerManager
class DirectInputDevice(base.Device):

    def __init__(self, display, device, device_instance):
        name = device_instance.tszInstanceName
        super(DirectInputDevice, self).__init__(display, name)
        self._type = device_instance.dwDevType & 255
        self._subtype = device_instance.dwDevType & 65280
        self._device = device
        self._init_controls()
        self._set_format()
        self.id_name = device_instance.tszProductName
        self.id_product_guid = format(device_instance.guidProduct.Data1, '08x')

    def __del__(self):
        self._device.Release()

    def get_guid(self):
        """Generate an SDL2 style GUID from the product guid."""
        first = self.id_product_guid[6:8] + self.id_product_guid[4:6]
        second = self.id_product_guid[2:4] + self.id_product_guid[0:2]
        return f'03000000{first}0000{second}000000000000'

    def _init_controls(self):
        self.controls = []
        self._device.EnumObjects(dinput.LPDIENUMDEVICEOBJECTSCALLBACK(self._object_enum), None, dinput.DIDFT_ALL)
        self.controls.sort(key=lambda c: c._type)

    def _object_enum(self, object_instance, arg):
        control = _create_control(object_instance.contents)
        if control:
            self.controls.append(control)
        return dinput.DIENUM_CONTINUE

    def _set_format(self):
        if not self.controls:
            return
        object_formats = (dinput.DIOBJECTDATAFORMAT * len(self.controls))()
        offset = 0
        for object_format, control in zip(object_formats, self.controls):
            object_format.dwOfs = offset
            object_format.dwType = control._type
            offset += 4
        fmt = dinput.DIDATAFORMAT()
        fmt.dwSize = ctypes.sizeof(fmt)
        fmt.dwObjSize = ctypes.sizeof(dinput.DIOBJECTDATAFORMAT)
        fmt.dwFlags = 0
        fmt.dwDataSize = offset
        fmt.dwNumObjs = len(object_formats)
        fmt.rgodf = ctypes.cast(ctypes.pointer(object_formats), dinput.LPDIOBJECTDATAFORMAT)
        self._device.SetDataFormat(fmt)
        prop = dinput.DIPROPDWORD()
        prop.diph.dwSize = ctypes.sizeof(prop)
        prop.diph.dwHeaderSize = ctypes.sizeof(prop.diph)
        prop.diph.dwObj = 0
        prop.diph.dwHow = dinput.DIPH_DEVICE
        prop.dwData = 64 * ctypes.sizeof(dinput.DIDATAFORMAT)
        self._device.SetProperty(dinput.DIPROP_BUFFERSIZE, ctypes.byref(prop.diph))

    def open(self, window=None, exclusive=False):
        if not self.controls:
            return
        if window is None:
            window = pyglet.gl._shadow_window
            for window in pyglet.app.windows:
                break
        flags = dinput.DISCL_BACKGROUND
        if exclusive:
            flags |= dinput.DISCL_EXCLUSIVE
        else:
            flags |= dinput.DISCL_NONEXCLUSIVE
        self._wait_object = _kernel32.CreateEventW(None, False, False, None)
        self._device.SetEventNotification(self._wait_object)
        pyglet.app.platform_event_loop.add_wait_object(self._wait_object, self._dispatch_events)
        self._device.SetCooperativeLevel(window._hwnd, flags)
        self._device.Acquire()

    def close(self):
        if not self.controls:
            return
        pyglet.app.platform_event_loop.remove_wait_object(self._wait_object)
        self._device.Unacquire()
        self._device.SetEventNotification(None)
        _kernel32.CloseHandle(self._wait_object)

    def get_controls(self):
        return self.controls

    def _dispatch_events(self):
        if not self.controls:
            return
        events = (dinput.DIDEVICEOBJECTDATA * 64)()
        n_events = win32.DWORD(len(events))
        try:
            self._device.GetDeviceData(ctypes.sizeof(dinput.DIDEVICEOBJECTDATA), ctypes.cast(ctypes.pointer(events), dinput.LPDIDEVICEOBJECTDATA), ctypes.byref(n_events), 0)
        except OSError:
            return
        for event in events[:n_events.value]:
            index = event.dwOfs // 4
            self.controls[index].value = event.dwData

    def matches(self, guid_id, device_instance):
        if self.id_product_guid == guid_id and self.id_name == device_instance.contents.tszProductName and (self._type == device_instance.contents.dwDevType & 255) and (self._subtype == device_instance.contents.dwDevType & 65280):
            return True
        return False