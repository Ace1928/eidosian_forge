from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class VmTagsValue(_messages.Message):
    """Optional. Resource manager tags to be bound to this instance. Tag keys
    and values have the same definition as https://cloud.google.com/resource-
    manager/docs/tags/tags-overview Keys must be in the format
    `tagKeys/{tag_key_id}`, and values are in the format `tagValues/456`.

    Messages:
      AdditionalProperty: An additional property for a VmTagsValue object.

    Fields:
      additionalProperties: Additional properties of type VmTagsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a VmTagsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)