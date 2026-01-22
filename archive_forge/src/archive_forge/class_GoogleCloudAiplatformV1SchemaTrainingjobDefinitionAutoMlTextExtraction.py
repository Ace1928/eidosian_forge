from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SchemaTrainingjobDefinitionAutoMlTextExtraction(_messages.Message):
    """A TrainingJob that trains and uploads an AutoML Text Extraction Model.

  Fields:
    inputs: The input parameters of this TrainingJob.
  """
    inputs = _messages.MessageField('GoogleCloudAiplatformV1SchemaTrainingjobDefinitionAutoMlTextExtractionInputs', 1)