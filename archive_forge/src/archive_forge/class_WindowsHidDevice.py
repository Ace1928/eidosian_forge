import ctypes
from ctypes import wintypes
import platform
from pyu2f import errors
from pyu2f.hid import base
class WindowsHidDevice(base.HidDevice):
    """Implementation of raw HID interface on Windows."""

    @staticmethod
    def Enumerate():
        """See base class."""
        hid_guid = GUID()
        hid.HidD_GetHidGuid(ctypes.byref(hid_guid))
        devices = setupapi.SetupDiGetClassDevsA(ctypes.byref(hid_guid), None, None, 18)
        index = 0
        interface_info = DeviceInterfaceData()
        interface_info.cbSize = ctypes.sizeof(DeviceInterfaceData)
        out = []
        while True:
            result = setupapi.SetupDiEnumDeviceInterfaces(devices, 0, ctypes.byref(hid_guid), index, ctypes.byref(interface_info))
            index += 1
            if not result:
                break
            detail_len = wintypes.DWORD()
            result = setupapi.SetupDiGetDeviceInterfaceDetailA(devices, ctypes.byref(interface_info), None, 0, ctypes.byref(detail_len), None)
            detail_len = detail_len.value
            if detail_len == 0:
                continue
            buf = ctypes.create_string_buffer(detail_len)
            interface_detail = DeviceInterfaceDetailData.from_buffer(buf)
            interface_detail.cbSize = ctypes.sizeof(DeviceInterfaceDetailData)
            result = setupapi.SetupDiGetDeviceInterfaceDetailA(devices, ctypes.byref(interface_info), ctypes.byref(interface_detail), detail_len, None, None)
            if not result:
                raise ctypes.WinError()
            descriptor = base.DeviceDescriptor()
            path_len = detail_len - ctypes.sizeof(wintypes.DWORD)
            descriptor.path = ctypes.string_at(ctypes.addressof(interface_detail.DevicePath), path_len)
            device = None
            try:
                device = OpenDevice(descriptor.path, True)
            except WindowsError as e:
                if e.winerror == ERROR_ACCESS_DENIED:
                    continue
                else:
                    raise e
            try:
                FillDeviceAttributes(device, descriptor)
                FillDeviceCapabilities(device, descriptor)
                out.append(descriptor.ToPublicDict())
            finally:
                kernel32.CloseHandle(device)
        return out

    def __init__(self, path):
        """See base class."""
        base.HidDevice.__init__(self, path)
        self.dev = OpenDevice(path)
        self.desc = base.DeviceDescriptor()
        FillDeviceCapabilities(self.dev, self.desc)

    def GetInReportDataLength(self):
        """See base class."""
        return self.desc.internal_max_in_report_len - 1

    def GetOutReportDataLength(self):
        """See base class."""
        return self.desc.internal_max_out_report_len - 1

    def Write(self, packet):
        """See base class."""
        if len(packet) != self.GetOutReportDataLength():
            raise errors.HidError('Packet length must match report data length.')
        packet_data = [0] + packet
        out = bytes(bytearray(packet_data))
        num_written = wintypes.DWORD()
        ret = kernel32.WriteFile(self.dev, out, len(out), ctypes.byref(num_written), None)
        if num_written.value != len(out):
            raise errors.HidError('Failed to write complete packet.  ' + 'Expected %d, but got %d' % (len(out), num_written.value))
        if not ret:
            raise ctypes.WinError()

    def Read(self):
        """See base class."""
        buf = ctypes.create_string_buffer(self.desc.internal_max_in_report_len)
        num_read = wintypes.DWORD()
        ret = kernel32.ReadFile(self.dev, buf, len(buf), ctypes.byref(num_read), None)
        if num_read.value != self.desc.internal_max_in_report_len:
            raise errors.HidError('Failed to read full length report from device.')
        if not ret:
            raise ctypes.WinError()
        return list(bytearray(buf[1:]))

    def __del__(self):
        """Closes the file handle when object is GC-ed."""
        if hasattr(self, 'dev'):
            kernel32.CloseHandle(self.dev)