from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsApplyRequest(_messages.Message):
    """A
  TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsApplyRequest
  object.

  Fields:
    applyDeploymentRequest: A ApplyDeploymentRequest resource to be passed as
      the request body.
    name: Required. The name of the deployment to apply to orchestration
      cluster.
  """
    applyDeploymentRequest = _messages.MessageField('ApplyDeploymentRequest', 1)
    name = _messages.StringField(2, required=True)