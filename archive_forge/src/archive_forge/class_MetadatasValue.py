from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class MetadatasValue(_messages.Message):
    """Additional structured details about this error. Keys must match /a-z+/
    but should ideally be lowerCamelCase. Also they must be limited to 64
    characters in length. When identifying the current value of an exceeded
    limit, the units should be contained in the key, not the value. For
    example, rather than {"instanceLimit": "100/request"}, should be returned
    as, {"instanceLimitPerRequest": "100"}, if the client exceeds the number
    of instances that can be created in a single (batch) request.

    Messages:
      AdditionalProperty: An additional property for a MetadatasValue object.

    Fields:
      additionalProperties: Additional properties of type MetadatasValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a MetadatasValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)