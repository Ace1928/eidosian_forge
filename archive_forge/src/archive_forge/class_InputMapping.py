from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InputMapping(_messages.Message):
    """InputMapping creates a 'virtual' property that will be injected into the
  properties before sending the request to the underlying API.

  Enums:
    LocationValueValuesEnum: The location where this mapping applies.

  Fields:
    fieldName: The name of the field that is going to be injected.
    location: The location where this mapping applies.
    methodMatch: Regex to evaluate on method to decide if input applies.
    value: A jsonPath expression to select an element.
  """

    class LocationValueValuesEnum(_messages.Enum):
        """The location where this mapping applies.

    Values:
      UNKNOWN: <no description>
      PATH: <no description>
      QUERY: <no description>
      BODY: <no description>
      HEADER: <no description>
    """
        UNKNOWN = 0
        PATH = 1
        QUERY = 2
        BODY = 3
        HEADER = 4
    fieldName = _messages.StringField(1)
    location = _messages.EnumField('LocationValueValuesEnum', 2)
    methodMatch = _messages.StringField(3)
    value = _messages.StringField(4)