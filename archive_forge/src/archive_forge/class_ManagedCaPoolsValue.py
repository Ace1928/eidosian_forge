from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class ManagedCaPoolsValue(_messages.Message):
    """A map from a region to the status of managed CA pools in that region.

    Messages:
      AdditionalProperty: An additional property for a ManagedCaPoolsValue
        object.

    Fields:
      additionalProperties: Additional properties of type ManagedCaPoolsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a ManagedCaPoolsValue object.

      Fields:
        key: Name of the additional property.
        value: A CaPoolsStatus attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('CaPoolsStatus', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)