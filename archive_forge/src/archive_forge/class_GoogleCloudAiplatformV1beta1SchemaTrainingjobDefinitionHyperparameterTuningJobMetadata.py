from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionHyperparameterTuningJobMetadata(_messages.Message):
    """A GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionHyperparameterT
  uningJobMetadata object.

  Fields:
    backingHyperparameterTuningJob: The resource name of the
      HyperparameterTuningJob that has been created to carry out this
      HyperparameterTuning task.
    bestTrialBackingCustomJob: The resource name of the CustomJob that has
      been created to run the best Trial of this HyperparameterTuning task.
  """
    backingHyperparameterTuningJob = _messages.StringField(1)
    bestTrialBackingCustomJob = _messages.StringField(2)