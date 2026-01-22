from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsTensorboardsExperimentsWriteRequest(_messages.Message):
    """A AiplatformProjectsLocationsTensorboardsExperimentsWriteRequest object.

  Fields:
    googleCloudAiplatformV1WriteTensorboardExperimentDataRequest: A
      GoogleCloudAiplatformV1WriteTensorboardExperimentDataRequest resource to
      be passed as the request body.
    tensorboardExperiment: Required. The resource name of the
      TensorboardExperiment to write data to. Format: `projects/{project}/loca
      tions/{location}/tensorboards/{tensorboard}/experiments/{experiment}`
  """
    googleCloudAiplatformV1WriteTensorboardExperimentDataRequest = _messages.MessageField('GoogleCloudAiplatformV1WriteTensorboardExperimentDataRequest', 1)
    tensorboardExperiment = _messages.StringField(2, required=True)