from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CustomErrorRule(_messages.Message):
    """A custom error rule.

  Fields:
    isErrorType: Mark this message as possible payload in error response.
      Otherwise, objects of this type will be filtered when they appear in
      error payload.
    selector: Selects messages to which this rule applies.  Refer to selector
      for syntax details.
  """
    isErrorType = _messages.BooleanField(1)
    selector = _messages.StringField(2)