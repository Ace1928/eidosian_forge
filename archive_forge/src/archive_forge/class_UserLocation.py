from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UserLocation(_messages.Message):
    """JSON template for a location entry.

  Fields:
    area: Textual location. This is most useful for display purposes to
      concisely describe the location. For example, "Mountain View, CA", "Near
      Seattle", "US-NYC-9TH 9A209A".
    buildingId: Building Identifier.
    customType: Custom Type.
    deskCode: Most specific textual code of individual desk location.
    floorName: Floor name/number.
    floorSection: Floor section. More specific location within the floor. For
      example, if a floor is divided into sections "A", "B", and "C", this
      field would identify one of those values.
    type: Each entry can have a type which indicates standard types of that
      entry. For example location could be of types default and desk. In
      addition to standard type, an entry can have a custom type and can give
      it any name. Such types should have "custom" as type and also have a
      customType value.
  """
    area = _messages.StringField(1)
    buildingId = _messages.StringField(2)
    customType = _messages.StringField(3)
    deskCode = _messages.StringField(4)
    floorName = _messages.StringField(5)
    floorSection = _messages.StringField(6)
    type = _messages.StringField(7)