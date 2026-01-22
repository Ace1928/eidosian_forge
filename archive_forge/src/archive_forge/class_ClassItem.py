from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClassItem(_messages.Message):
    """An item of the class.

  Fields:
    value: The class item's value.
  """
    value = _messages.StringField(1)