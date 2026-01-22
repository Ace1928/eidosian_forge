from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VersionSource(_messages.Message):
    """Describes a selector for extracting and matching an MSH field to a
  value.

  Fields:
    mshField: The field to extract from the MSH segment. For example, "3.1" or
      "18[1].1".
    value: The value to match with the field. For example, "My Application
      Name" or "2.3".
  """
    mshField = _messages.StringField(1)
    value = _messages.StringField(2)