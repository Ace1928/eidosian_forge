from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1NasJobOutputMultiTrialJobOutputNasParameterMetric(_messages.Message):
    """An observed value of a metric of the trial.

  Messages:
    MetricsValue: Reported metrics other than objective and model_flops

  Fields:
    metrics: Reported metrics other than objective and model_flops
    modelFlops: The model flops associated with the `objective_value`.
    objectiveValue: The objective value at this training step.
    trainingStep: The global training step for this metric.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class MetricsValue(_messages.Message):
        """Reported metrics other than objective and model_flops

    Messages:
      AdditionalProperty: An additional property for a MetricsValue object.

    Fields:
      additionalProperties: Additional properties of type MetricsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a MetricsValue object.

      Fields:
        key: Name of the additional property.
        value: A number attribute.
      """
            key = _messages.StringField(1)
            value = _messages.FloatField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    metrics = _messages.MessageField('MetricsValue', 1)
    modelFlops = _messages.FloatField(2)
    objectiveValue = _messages.FloatField(3)
    trainingStep = _messages.IntegerField(4)