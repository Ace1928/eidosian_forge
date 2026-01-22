from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QueryInfo(_messages.Message):
    """Query optimization information for a QUERY job.

  Messages:
    OptimizationDetailsValue: Output only. Information about query
      optimizations.

  Fields:
    optimizationDetails: Output only. Information about query optimizations.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class OptimizationDetailsValue(_messages.Message):
        """Output only. Information about query optimizations.

    Messages:
      AdditionalProperty: An additional property for a
        OptimizationDetailsValue object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a OptimizationDetailsValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    optimizationDetails = _messages.MessageField('OptimizationDetailsValue', 1)