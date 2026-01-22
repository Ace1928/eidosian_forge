from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class PerLocationStatusValue(_messages.Message):
    """Status information per location (location name is key). Example key:
    zones/us-central1-a

    Messages:
      AdditionalProperty: An additional property for a PerLocationStatusValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        PerLocationStatusValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a PerLocationStatusValue object.

      Fields:
        key: Name of the additional property.
        value: A BulkInsertOperationStatus attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('BulkInsertOperationStatus', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)