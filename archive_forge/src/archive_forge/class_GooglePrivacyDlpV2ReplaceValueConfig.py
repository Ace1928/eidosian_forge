from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2ReplaceValueConfig(_messages.Message):
    """Replace each input value with a given `Value`.

  Fields:
    newValue: Value to replace it with.
  """
    newValue = _messages.MessageField('GooglePrivacyDlpV2Value', 1)