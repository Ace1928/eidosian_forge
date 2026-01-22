from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HelloWorldFooBar(_messages.Message):
    """Nested Message.

  Fields:
    first: A string attribute.
    second: A integer attribute.
  """
    first = _messages.StringField(1)
    second = _messages.IntegerField(2, variant=_messages.Variant.INT32)