from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CustomPlacementConfigValue(_messages.Message):
    """The bucket's custom placement configuration for Custom Dual Regions.

    Fields:
      dataLocations: The list of regional locations in which data is placed.
    """
    dataLocations = _messages.StringField(1, repeated=True)