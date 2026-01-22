from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class EntityMetricsValue(_messages.Message):
    """Metrics across confidence levels, for different entities.

    Messages:
      AdditionalProperty: An additional property for a EntityMetricsValue
        object.

    Fields:
      additionalProperties: Additional properties of type EntityMetricsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a EntityMetricsValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudDocumentaiV1EvaluationMultiConfidenceMetrics
          attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('GoogleCloudDocumentaiV1EvaluationMultiConfidenceMetrics', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)