import ctypes
from ctypes import wintypes
import platform
from pyu2f import errors
from pyu2f.hid import base
def FillDeviceAttributes(device, descriptor):
    """Fill out the attributes of the device.

  Fills the devices HidAttributes and product string
  into the descriptor.

  Args:
    device: A handle to the open device
    descriptor: The DeviceDescriptor to populate with the
      attributes.

  Returns:
    None

  Raises:
    WindowsError when unable to obtain attributes or product
      string.
  """
    attributes = HidAttributes()
    result = hid.HidD_GetAttributes(device, ctypes.byref(attributes))
    if not result:
        raise ctypes.WinError()
    buf = ctypes.create_string_buffer(1024)
    result = hid.HidD_GetProductString(device, buf, 1024)
    if not result:
        raise ctypes.WinError()
    descriptor.vendor_id = attributes.VendorID
    descriptor.product_id = attributes.ProductID
    descriptor.product_string = ctypes.wstring_at(buf)