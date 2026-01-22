from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class ExecutorSummaryValue(_messages.Message):
    """A ExecutorSummaryValue object.

    Messages:
      AdditionalProperty: An additional property for a ExecutorSummaryValue
        object.

    Fields:
      additionalProperties: Additional properties of type ExecutorSummaryValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a ExecutorSummaryValue object.

      Fields:
        key: Name of the additional property.
        value: A ExecutorStageSummary attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('ExecutorStageSummary', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)