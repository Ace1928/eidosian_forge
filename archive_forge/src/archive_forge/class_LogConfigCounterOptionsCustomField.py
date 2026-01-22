from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LogConfigCounterOptionsCustomField(_messages.Message):
    """Custom fields. These can be used to create a counter with arbitrary
  field/value pairs. See: go/rpcsp-custom-fields.

  Fields:
    name: Name is the field name.
    value: Value is the field value. It is important that in contrast to the
      CounterOptions.field, the value here is a constant that is not derived
      from the IAMContext.
  """
    name = _messages.StringField(1)
    value = _messages.StringField(2)