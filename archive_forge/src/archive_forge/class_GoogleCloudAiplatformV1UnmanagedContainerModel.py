from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1UnmanagedContainerModel(_messages.Message):
    """Contains model information necessary to perform batch prediction without
  requiring a full model import.

  Fields:
    artifactUri: The path to the directory containing the Model artifact and
      any of its supporting files.
    containerSpec: Input only. The specification of the container that is to
      be used when deploying this Model.
    predictSchemata: Contains the schemata used in Model's predictions and
      explanations
  """
    artifactUri = _messages.StringField(1)
    containerSpec = _messages.MessageField('GoogleCloudAiplatformV1ModelContainerSpec', 2)
    predictSchemata = _messages.MessageField('GoogleCloudAiplatformV1PredictSchemata', 3)