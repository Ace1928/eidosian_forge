from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicytroubleshooterIamV3ConditionContextRequest(_messages.Message):
    """This message defines attributes for an HTTP request. If the actual
  request is not an HTTP request, the runtime system should try to map the
  actual request to an equivalent HTTP request.

  Fields:
    receiveTime: Optional. The timestamp when the destination service receives
      the first byte of the request.
  """
    receiveTime = _messages.StringField(1)