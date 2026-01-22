from __future__ import division
import os
import struct
from pyu2f import errors
from pyu2f.hid import base
class LinuxHidDevice(base.HidDevice):
    """Implementation of HID device for linux.

  Implementation of HID device interface for linux that uses block
  devices to interact with the device and sysfs to enumerate/discover
  device metadata.
  """

    @staticmethod
    def Enumerate():
        for hidraw in os.listdir('/sys/class/hidraw'):
            rd_path = os.path.join('/sys/class/hidraw', hidraw, 'device/report_descriptor')
            uevent_path = os.path.join('/sys/class/hidraw', hidraw, 'device/uevent')
            rd_file = open(rd_path, 'rb')
            uevent_file = open(uevent_path, 'rb')
            desc = base.DeviceDescriptor()
            desc.path = os.path.join('/dev/', hidraw)
            ParseReportDescriptor(rd_file.read(), desc)
            ParseUevent(uevent_file.read(), desc)
            rd_file.close()
            uevent_file.close()
            yield desc.ToPublicDict()

    def __init__(self, path):
        base.HidDevice.__init__(self, path)
        self.dev = os.open(path, os.O_RDWR)
        self.desc = base.DeviceDescriptor()
        self.desc.path = path
        rd_file = open(os.path.join('/sys/class/hidraw', os.path.basename(path), 'device/report_descriptor'), 'rb')
        ParseReportDescriptor(rd_file.read(), self.desc)
        rd_file.close()

    def GetInReportDataLength(self):
        """See base class."""
        return self.desc.internal_max_in_report_len

    def GetOutReportDataLength(self):
        """See base class."""
        return self.desc.internal_max_out_report_len

    def Write(self, packet):
        """See base class."""
        out = bytearray([0] + packet)
        os.write(self.dev, out)

    def Read(self):
        """See base class."""
        raw_in = os.read(self.dev, self.GetInReportDataLength())
        decoded_in = list(bytearray(raw_in))
        return decoded_in