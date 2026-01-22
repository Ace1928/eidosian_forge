from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class ComponentStatesValue(_messages.Message):
    """Currently these include (also serving as map keys): 1. "admission" 2.
    "audit" 3. "mutation"

    Messages:
      AdditionalProperty: An additional property for a ComponentStatesValue
        object.

    Fields:
      additionalProperties: Additional properties of type ComponentStatesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a ComponentStatesValue object.

      Fields:
        key: Name of the additional property.
        value: A PolicyControllerOnClusterState attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('PolicyControllerOnClusterState', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)