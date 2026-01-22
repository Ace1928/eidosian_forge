from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasportalCustomersNodesDevicesListRequest(_messages.Message):
    """A SasportalCustomersNodesDevicesListRequest object.

  Fields:
    filter: The filter expression. The filter should have one of the following
      formats: "sn=123454" or "display_name=MyDevice". sn corresponds to
      serial number of the device. The filter is case insensitive.
    pageSize: The maximum number of devices to return in the response. If
      empty or zero, all devices will be listed. Must be in the range [0,
      1000].
    pageToken: A pagination token returned from a previous call to ListDevices
      that indicates where this listing should continue from.
    parent: Required. The name of the parent resource.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)