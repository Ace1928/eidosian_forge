from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3DeployFlowResponse(_messages.Message):
    """The response message for Environments.DeployFlow.

  Fields:
    deployment: The name of the flow version Deployment. Format:
      `projects//locations//agents// environments//deployments/`.
    environment: The updated environment where the flow is deployed.
  """
    deployment = _messages.StringField(1)
    environment = _messages.MessageField('GoogleCloudDialogflowCxV3Environment', 2)