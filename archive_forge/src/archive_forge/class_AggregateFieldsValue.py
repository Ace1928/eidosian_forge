from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class AggregateFieldsValue(_messages.Message):
    """The result of the aggregation functions, ex: `COUNT(*) AS total_docs`.
    The key is the alias assigned to the aggregation function on input and the
    size of this map equals the number of aggregation functions in the query.

    Messages:
      AdditionalProperty: An additional property for a AggregateFieldsValue
        object.

    Fields:
      additionalProperties: Additional properties of type AggregateFieldsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a AggregateFieldsValue object.

      Fields:
        key: Name of the additional property.
        value: A Value attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('Value', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)