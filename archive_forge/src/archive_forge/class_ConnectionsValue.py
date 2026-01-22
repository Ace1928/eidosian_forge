from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class ConnectionsValue(_messages.Message):
    """Optional. A mapping from connection ID to Connections to make
    available to the workflow. Each map entry's key should match the `id`
    field of the // value.

    Messages:
      AdditionalProperty: An additional property for a ConnectionsValue
        object.

    Fields:
      additionalProperties: Additional properties of type ConnectionsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a ConnectionsValue object.

      Fields:
        key: Name of the additional property.
        value: A Connection attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('Connection', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)