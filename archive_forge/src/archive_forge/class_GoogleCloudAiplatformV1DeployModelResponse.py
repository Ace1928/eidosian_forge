from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1DeployModelResponse(_messages.Message):
    """Response message for EndpointService.DeployModel.

  Fields:
    deployedModel: The DeployedModel that had been deployed in the Endpoint.
  """
    deployedModel = _messages.MessageField('GoogleCloudAiplatformV1DeployedModel', 1)