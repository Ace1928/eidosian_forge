from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsPatchRequest(_messages.Message):
    """A
  TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsPatchRequest
  object.

  Fields:
    deployment: A Deployment resource to be passed as the request body.
    name: The name of the deployment.
    updateMask: Required. Update mask is used to specify the fields to be
      overwritten in the `deployment` resource by the update.
  """
    deployment = _messages.MessageField('Deployment', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)