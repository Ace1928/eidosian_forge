from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RouterStatus(_messages.Message):
    """Detail Status of Router resource.

  Fields:
    ipAddress: IP Address of the Google Cloud Load Balancer.
  """
    ipAddress = _messages.StringField(1)