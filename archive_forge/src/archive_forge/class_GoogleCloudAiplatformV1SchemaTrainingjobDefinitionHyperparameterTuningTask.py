from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SchemaTrainingjobDefinitionHyperparameterTuningTask(_messages.Message):
    """A TrainingJob that tunes Hypererparameters of a custom code Model.

  Fields:
    inputs: The input parameters of this HyperparameterTuningTask.
    metadata: The metadata information.
  """
    inputs = _messages.MessageField('GoogleCloudAiplatformV1SchemaTrainingjobDefinitionHyperparameterTuningJobSpec', 1)
    metadata = _messages.MessageField('GoogleCloudAiplatformV1SchemaTrainingjobDefinitionHyperparameterTuningJobMetadata', 2)