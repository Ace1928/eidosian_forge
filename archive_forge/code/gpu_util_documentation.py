import collections
import re
Returns the GpuInfo given a DeviceAttributes proto.

  Args:
    device_attrs: A DeviceAttributes proto.

  Returns
    A gpu_info tuple. Both fields are None if `device_attrs` does not have a
    valid physical_device_desc field.
  