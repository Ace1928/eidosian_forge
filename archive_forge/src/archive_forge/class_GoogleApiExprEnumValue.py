from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiExprEnumValue(_messages.Message):
    """An enum value.

  Fields:
    type: The fully qualified name of the enum type.
    value: The value of the enum.
  """
    type = _messages.StringField(1)
    value = _messages.IntegerField(2, variant=_messages.Variant.INT32)