from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class SubqueriesValue(_messages.Message):
    """A mapping of (subquery variable name) -> (subquery node id) for cases
    where the `description` string of this node references a `SCALAR` subquery
    contained in the expression subtree rooted at this node. The referenced
    `SCALAR` subquery may not necessarily be a direct child of this node.

    Messages:
      AdditionalProperty: An additional property for a SubqueriesValue object.

    Fields:
      additionalProperties: Additional properties of type SubqueriesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a SubqueriesValue object.

      Fields:
        key: Name of the additional property.
        value: A integer attribute.
      """
        key = _messages.StringField(1)
        value = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)