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
class DIDeviceManager(EventDispatcher):

    def __init__(self):
        self.registered = False
        self.window = None
        self._devnotify = None
        self.devices: List[DirectInputDevice] = []
        if self.register_device_events(skip_warning=True):
            self.set_current_devices()

    def set_current_devices(self):
        """Sets all currently discovered devices in the devices list.
        Be careful when using this, as this creates new device objects. Should only be called on initialization of
        the manager and if for some reason the window connection event isn't registered.
        """
        new_devices, _ = self._get_devices()
        self.devices = new_devices

    def register_device_events(self, skip_warning=False, window=None):
        """Register the first OS Window to receive events of disconnect and connection of devices.
        Returns True if events were successfully registered.
        """
        if not self.registered:
            if not window:
                window = pyglet.gl._shadow_window
                if not window:
                    for window in pyglet.app.windows:
                        break
            self.window = window
            if self.window is not None:
                dbi = DEV_BROADCAST_DEVICEINTERFACE()
                dbi.dbcc_size = ctypes.sizeof(dbi)
                dbi.dbcc_devicetype = DBT_DEVTYP_DEVICEINTERFACE
                dbi.dbcc_classguid = GUID_DEVINTERFACE_HID
                self._devnotify = _user32.RegisterDeviceNotificationW(self.window._hwnd, ctypes.byref(dbi), DEVICE_NOTIFY_WINDOW_HANDLE)
                self.window._event_handlers[WM_DEVICECHANGE] = self._event_devicechange
                self.registered = True
                self.window.push_handlers(self)
                return True
            elif not skip_warning:
                warnings.warn('DirectInput Device Manager requires a window to receive device connection events.')
        return False

    def _unregister_device_events(self):
        del self.window._event_handlers[WM_DEVICECHANGE]
        _user32.UnregisterDeviceNotification(self._devnotify)
        self.registered = False
        self._devnotify = None

    def on_close(self):
        if self.registered:
            self._unregister_device_events()
        import pyglet.app
        if len(pyglet.app.windows) != 0:
            for existing_window in pyglet.app.windows:
                if existing_window != self.window:
                    self.register_device_events(skip_warning=True, window=existing_window)
                    return
        self.window = None

    def __del__(self):
        if self.registered:
            self._unregister_device_events()

    def _get_devices(self, display=None):
        """Enumerate all the devices on the system.
        Returns two values: new devices, missing devices"""
        _missing_devices = list(self.devices)
        _new_devices = []
        _xinput_devices = []
        if not pyglet.options['win32_disable_xinput']:
            try:
                from pyglet.input.win32.xinput import get_xinput_guids
                _xinput_devices = get_xinput_guids()
            except ImportError:
                pass

        def _device_enum(device_instance, arg):
            guid_id = format(device_instance.contents.guidProduct.Data1, '08x')
            if guid_id in _xinput_devices:
                return dinput.DIENUM_CONTINUE
            for dev in list(_missing_devices):
                if dev.matches(guid_id, device_instance):
                    _missing_devices.remove(dev)
                    return dinput.DIENUM_CONTINUE
            device = dinput.IDirectInputDevice8()
            _i_dinput.CreateDevice(device_instance.contents.guidInstance, ctypes.byref(device), None)
            di_dev = DirectInputDevice(display, device, device_instance.contents)
            _new_devices.append(di_dev)
            return dinput.DIENUM_CONTINUE
        _i_dinput.EnumDevices(dinput.DI8DEVCLASS_ALL, dinput.LPDIENUMDEVICESCALLBACK(_device_enum), None, dinput.DIEDFL_ATTACHEDONLY)
        return (_new_devices, _missing_devices)

    def _recheck_devices(self):
        new_devices, missing_devices = self._get_devices()
        if new_devices:
            self.devices.extend(new_devices)
            for device in new_devices:
                self.dispatch_event('on_connect', device)
        if missing_devices:
            for device in missing_devices:
                self.devices.remove(device)
                self.dispatch_event('on_disconnect', device)

    def _event_devicechange(self, msg, wParam, lParam):
        if lParam == 0:
            return
        if wParam == DBT_DEVICEARRIVAL or wParam == DBT_DEVICEREMOVECOMPLETE:
            hdr_ptr = ctypes.cast(lParam, ctypes.POINTER(DEV_BROADCAST_HDR))
            if hdr_ptr.contents.dbch_devicetype == DBT_DEVTYP_DEVICEINTERFACE:
                pyglet.app.platform_event_loop.post_event(self, '_recheck_devices')