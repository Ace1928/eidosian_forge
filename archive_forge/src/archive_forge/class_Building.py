from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Building(_messages.Message):
    """JSON template for Building object in Directory API.

  Fields:
    address: The postal address of the building. See PostalAddress for
      details. Note that only a single address line and region code are
      required.
    buildingId: Unique identifier for the building. The maximum length is 100
      characters.
    buildingName: The building name as seen by users in Calendar. Must be
      unique for the customer. For example, "NYC-CHEL". The maximum length is
      100 characters.
    coordinates: The geographic coordinates of the center of the building,
      expressed as latitude and longitude in decimal degrees.
    description: A brief description of the building. For example, "Chelsea
      Market".
    etags: ETag of the resource.
    floorNames: The display names for all floors in this building. The floors
      are expected to be sorted in ascending order, from lowest floor to
      highest floor. For example, ["B2", "B1", "L", "1", "2", "2M", "3", "PH"]
      Must contain at least one entry.
    kind: Kind of resource this is.
  """
    address = _messages.MessageField('BuildingAddress', 1)
    buildingId = _messages.StringField(2)
    buildingName = _messages.StringField(3)
    coordinates = _messages.MessageField('BuildingCoordinates', 4)
    description = _messages.StringField(5)
    etags = _messages.StringField(6)
    floorNames = _messages.StringField(7, repeated=True)
    kind = _messages.StringField(8, default=u'admin#directory#resources#buildings#Building')