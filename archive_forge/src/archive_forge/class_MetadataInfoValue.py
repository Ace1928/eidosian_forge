from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class MetadataInfoValue(_messages.Message):
    """Output only. System-generated information about the metadata fields.
    Includes update time and owner.

    Messages:
      AdditionalProperty: An additional property for a MetadataInfoValue
        object.

    Fields:
      additionalProperties: Additional properties of type MetadataInfoValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a MetadataInfoValue object.

      Fields:
        key: Name of the additional property.
        value: A MetadataInfo attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('MetadataInfo', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)