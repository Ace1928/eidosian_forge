from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExecutorMetrics(_messages.Message):
    """A ExecutorMetrics object.

  Messages:
    MetricsValue: A MetricsValue object.

  Fields:
    metrics: A MetricsValue attribute.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class MetricsValue(_messages.Message):
        """A MetricsValue object.

    Messages:
      AdditionalProperty: An additional property for a MetricsValue object.

    Fields:
      additionalProperties: Additional properties of type MetricsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a MetricsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.IntegerField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    metrics = _messages.MessageField('MetricsValue', 1)