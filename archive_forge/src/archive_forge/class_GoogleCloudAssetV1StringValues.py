from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAssetV1StringValues(_messages.Message):
    """The string values for the list constraints.

  Fields:
    allowedValues: List of values allowed at this resource.
    deniedValues: List of values denied at this resource.
  """
    allowedValues = _messages.StringField(1, repeated=True)
    deniedValues = _messages.StringField(2, repeated=True)