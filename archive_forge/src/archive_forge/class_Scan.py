from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Scan(_messages.Message):
    """Scan is a structure which describes Cloud Key Visualizer scan
  information.

  Messages:
    DetailsValue: Additional information provided by the implementer.

  Fields:
    details: Additional information provided by the implementer.
    endTime: The upper bound for when the scan is defined.
    name: The unique name of the scan, specific to the Database service
      implementing this interface.
    scanData: Output only. Cloud Key Visualizer scan data. Note, this field is
      not available to the ListScans method.
    startTime: A range of time (inclusive) for when the scan is defined. The
      lower bound for when the scan is defined.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class DetailsValue(_messages.Message):
        """Additional information provided by the implementer.

    Messages:
      AdditionalProperty: An additional property for a DetailsValue object.

    Fields:
      additionalProperties: Properties of the object. Contains field @type
        with type URL.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a DetailsValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    details = _messages.MessageField('DetailsValue', 1)
    endTime = _messages.StringField(2)
    name = _messages.StringField(3)
    scanData = _messages.MessageField('ScanData', 4)
    startTime = _messages.StringField(5)