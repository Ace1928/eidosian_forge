from typing import List, Optional, Tuple
from pyglet.libs.win32 import com
from pyglet.libs.win32 import _ole32 as ole32
from pyglet.libs.win32.constants import CLSCTX_INPROC_SERVER
from pyglet.libs.win32.types import *
from pyglet.media.devices import base
from pyglet.util import debug_print
class Win32AudioDeviceManager(base.AbstractAudioDeviceManager):

    def __init__(self):
        self._device_enum = IMMDeviceEnumerator()
        ole32.CoCreateInstance(CLSID_MMDeviceEnumerator, None, CLSCTX_INPROC_SERVER, IID_IMMDeviceEnumerator, byref(self._device_enum))
        self.devices: List[Win32AudioDevice] = self._query_all_devices()
        super().__init__()
        self._callback = AudioNotificationCB(self)
        self._device_enum.RegisterEndpointNotificationCallback(self._callback)

    def add_device(self, pwstrDeviceId: str) -> Win32AudioDevice:
        dev = self.get_device(pwstrDeviceId)
        self.devices.append(dev)
        return dev

    def remove_device(self, pwstrDeviceId: str) -> Win32AudioDevice:
        dev = self.audio_devices.get_cached_device(pwstrDeviceId)
        self.audio_devices.devices.remove(dev)
        return dev

    def get_device(self, pwstrDeviceId: str) -> Win32AudioDevice:
        device = IMMDevice()
        self._device_enum.GetDevice(pwstrDeviceId, byref(device))
        dev_id, name, desc, dev_state = self.get_device_info(device)
        ep = IMMEndpoint()
        device.QueryInterface(IID_IMMEndpoint, byref(ep))
        dataflow = EDataFlow()
        ep.GetDataFlow(byref(dataflow))
        flow = dataflow.value
        windevice = Win32AudioDevice(dev_id, name, desc, flow, dev_state)
        ep.Release()
        device.Release()
        return windevice

    def get_default_output(self) -> Optional[Win32AudioDevice]:
        """Attempts to retrieve a default audio output for the system. Returns None if no available devices found."""
        try:
            device = IMMDevice()
            self._device_enum.GetDefaultAudioEndpoint(eRender, eConsole, byref(device))
            dev_id, name, desc, dev_state = self.get_device_info(device)
            device.Release()
            cached_dev = self.get_cached_device(dev_id)
            cached_dev.state = dev_state
            return cached_dev
        except OSError as err:
            assert _debug(f'No default audio output was found. {err}')
            return None

    def get_default_input(self) -> Optional[Win32AudioDevice]:
        """Attempts to retrieve a default audio input for the system. Returns None if no available devices found."""
        try:
            device = IMMDevice()
            self._device_enum.GetDefaultAudioEndpoint(eCapture, eConsole, byref(device))
            dev_id, name, desc, dev_state = self.get_device_info(device)
            device.Release()
            cached_dev = self.get_cached_device(dev_id)
            cached_dev.state = dev_state
            return cached_dev
        except OSError as err:
            assert _debug(f'No default input output was found. {err}')
            return None

    def get_cached_device(self, dev_id) -> Win32AudioDevice:
        """Gets the cached devices, so we can reduce calls to COM and tell current state vs new states."""
        for device in self.devices:
            if device.id == dev_id:
                return device
        raise Exception('Attempted to get a device that does not exist.', dev_id)

    def get_output_devices(self, state=DEVICE_STATE_ACTIVE) -> List[Win32AudioDevice]:
        return [device for device in self.devices if device.state == state and device.flow == eRender]

    def get_input_devices(self, state=DEVICE_STATE_ACTIVE) -> List[Win32AudioDevice]:
        return [device for device in self.devices if device.state == state and device.flow == eCapture]

    def get_all_devices(self) -> List[Win32AudioDevice]:
        return self.devices

    def _query_all_devices(self) -> List[Win32AudioDevice]:
        return self.get_devices(flow=eRender, state=DEVICE_STATEMASK_ALL) + self.get_devices(flow=eCapture, state=DEVICE_STATEMASK_ALL)

    def get_device_info(self, device: IMMDevice) -> Tuple[str, str, str, int]:
        """Return the ID, Name, and Description of the Audio Device."""
        store = IPropertyStore()
        device.OpenPropertyStore(STGM_READ, byref(store))
        dev_id = LPWSTR()
        device.GetId(byref(dev_id))
        name = self.get_pkey_value(store, PKEY_Device_FriendlyName)
        description = self.get_pkey_value(store, PKEY_Device_DeviceDesc)
        state = DWORD()
        device.GetState(byref(state))
        store.Release()
        return (dev_id.value, name, description, state.value)

    def get_devices(self, flow=eRender, state=DEVICE_STATE_ACTIVE):
        """Get's all of the specified devices (by default, all output and active)."""
        collection = IMMDeviceCollection()
        self._device_enum.EnumAudioEndpoints(flow, state, byref(collection))
        count = UINT()
        collection.GetCount(byref(count))
        devices = []
        for i in range(count.value):
            dev_itf = IMMDevice()
            collection.Item(i, byref(dev_itf))
            dev_id, name, desc, dev_state = self.get_device_info(dev_itf)
            device = Win32AudioDevice(dev_id, name, desc, flow, dev_state)
            dev_itf.Release()
            devices.append(device)
        collection.Release()
        return devices

    @staticmethod
    def get_pkey_value(store: IPropertyStore, pkey: PROPERTYKEY):
        try:
            propvar = PROPVARIANT()
            store.GetValue(pkey, byref(propvar))
            value = propvar.pwszVal
            ole32.PropVariantClear(byref(propvar))
        except Exception:
            value = 'Unknown'
        return value