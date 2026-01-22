from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3beta1DtmfInput(_messages.Message):
    """Represents the input for dtmf event.

  Fields:
    digits: The dtmf digits.
    finishDigit: The finish digit (if any).
  """
    digits = _messages.StringField(1)
    finishDigit = _messages.StringField(2)