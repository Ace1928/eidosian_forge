from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigProjectsLocationsDeploymentsUnlockRequest(_messages.Message):
    """A ConfigProjectsLocationsDeploymentsUnlockRequest object.

  Fields:
    name: Required. The name of the deployment in the format:
      'projects/{project_id}/locations/{location}/deployments/{deployment}'.
    unlockDeploymentRequest: A UnlockDeploymentRequest resource to be passed
      as the request body.
  """
    name = _messages.StringField(1, required=True)
    unlockDeploymentRequest = _messages.MessageField('UnlockDeploymentRequest', 2)