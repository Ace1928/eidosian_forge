from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1StudySpecParameterSpecCategoricalValueSpec(_messages.Message):
    """Value specification for a parameter in `CATEGORICAL` type.

  Fields:
    defaultValue: A default value for a `CATEGORICAL` parameter that is
      assumed to be a relatively good starting point. Unset value signals that
      there is no offered starting point. Currently only supported by the
      Vertex AI Vizier service. Not supported by HyperparameterTuningJob or
      TrainingPipeline.
    values: Required. The list of possible categories.
  """
    defaultValue = _messages.StringField(1)
    values = _messages.StringField(2, repeated=True)