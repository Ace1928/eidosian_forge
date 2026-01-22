from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2ProfileStatus(_messages.Message):
    """Success or errors for the profile generation.

  Fields:
    status: Profiling status code and optional message. The `status.code`
      value is 0 (default value) for OK.
    timestamp: Time when the profile generation status was updated
  """
    status = _messages.MessageField('GoogleRpcStatus', 1)
    timestamp = _messages.StringField(2)