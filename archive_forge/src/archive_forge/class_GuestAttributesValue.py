from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class GuestAttributesValue(_messages.Message):
    """Output only. The Compute Engine guest attributes. (see [Project and
    instance guest attributes](https://cloud.google.com/compute/docs/storing-
    retrieving-metadata#guest_attributes)).

    Messages:
      AdditionalProperty: An additional property for a GuestAttributesValue
        object.

    Fields:
      additionalProperties: Additional properties of type GuestAttributesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a GuestAttributesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)