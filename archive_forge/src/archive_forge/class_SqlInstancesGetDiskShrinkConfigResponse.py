from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlInstancesGetDiskShrinkConfigResponse(_messages.Message):
    """Instance get disk shrink config response.

  Fields:
    kind: This is always `sql#getDiskShrinkConfig`.
    message: Additional message to customers.
    minimalTargetSizeGb: The minimum size to which a disk can be shrunk in
      GigaBytes.
  """
    kind = _messages.StringField(1)
    message = _messages.StringField(2)
    minimalTargetSizeGb = _messages.IntegerField(3)