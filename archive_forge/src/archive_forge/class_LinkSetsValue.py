from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class LinkSetsValue(_messages.Message):
    """Named collections of Link Sets, each having qualified links.

    Messages:
      AdditionalProperty: An additional property for a LinkSetsValue object.

    Fields:
      additionalProperties: Additional properties of type LinkSetsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a LinkSetsValue object.

      Fields:
        key: Name of the additional property.
        value: A LinkSet attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('LinkSet', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)