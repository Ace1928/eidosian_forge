from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1StudySpecParameterSpecIntegerValueSpec(_messages.Message):
    """Value specification for a parameter in `INTEGER` type.

  Fields:
    defaultValue: A default value for an `INTEGER` parameter that is assumed
      to be a relatively good starting point. Unset value signals that there
      is no offered starting point. Currently only supported by the Vertex AI
      Vizier service. Not supported by HyperparameterTuningJob or
      TrainingPipeline.
    maxValue: Required. Inclusive maximum value of the parameter.
    minValue: Required. Inclusive minimum value of the parameter.
  """
    defaultValue = _messages.IntegerField(1)
    maxValue = _messages.IntegerField(2)
    minValue = _messages.IntegerField(3)