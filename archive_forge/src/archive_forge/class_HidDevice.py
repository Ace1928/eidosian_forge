class HidDevice(object):
    """Base class for all HID devices in this package."""

    @staticmethod
    def Enumerate():
        """Enumerates all the hid devices.

    This function enumerates all the hid device and provides metadata
    for helping the client select one.

    Returns:
      A list of dictionaries of metadata.  Each implementation is required
      to provide at least: vendor_id, product_id, product_string, usage,
      usage_page, and path.
    """
        pass

    def __init__(self, path):
        """Initialize the device at path."""
        pass

    def GetInReportDataLength(self):
        """Returns the max input report data length in bytes.

    Returns the max input report data length in bytes.  This excludes the
    report id.
    """
        pass

    def GetOutReportDataLength(self):
        """Returns the max output report data length in bytes.

    Returns the max output report data length in bytes.  This excludes the
    report id.
    """
        pass

    def Write(self, packet):
        """Writes packet to device.

    Writes the packet to the device.

    Args:
      packet: An array of integers to write to the device.  Excludes the report
      ID. Must be equal to GetOutReportLength().
    """
        pass

    def Read(self):
        """Reads packet from device.

    Reads the packet from the device.

    Returns:
      An array of integers read from the device.  Excludes the report ID.
      The length is equal to GetInReportDataLength().
    """
        pass