from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GetOwnerInstanceResponse(_messages.Message):
    """A GetOwnerInstanceResponse object.

  Fields:
    instance: Full instance resource URL.
  """
    instance = _messages.StringField(1)