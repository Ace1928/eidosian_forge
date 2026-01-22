from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class NodeTypeConfigsValue(_messages.Message):
    """Required. The map of cluster node types in this cluster, where the key
    is canonical identifier of the node type (corresponds to the `NodeType`).

    Messages:
      AdditionalProperty: An additional property for a NodeTypeConfigsValue
        object.

    Fields:
      additionalProperties: Additional properties of type NodeTypeConfigsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a NodeTypeConfigsValue object.

      Fields:
        key: Name of the additional property.
        value: A NodeTypeConfig attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('NodeTypeConfig', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)