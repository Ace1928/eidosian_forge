from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1DeploymentChangeReport(_messages.Message):
    """Response for GenerateDeployChangeReport and
  GenerateUndeployChangeReport. This report contains any validation failures
  that would cause the deployment to be rejected, as well changes and
  conflicts in routing that may occur due to the new deployment. The existence
  of a routing warning does not necessarily imply that the deployment request
  is bad, if the desired state of the deployment request is to effect a
  routing change. The primary purposes of the routing messages are: 1) To
  inform users of routing changes that may have an effect on traffic currently
  being routed to other existing deployments. 2) To warn users if some base
  path in the proxy will not receive traffic due to an existing deployment
  having already claimed that base path. The presence of routing
  conflicts/changes will not cause non-dry-run DeployApiProxy/UndeployApiProxy
  requests to be rejected.

  Fields:
    routingChanges: All routing changes that may result from a deployment
      request.
    routingConflicts: All base path conflicts detected for a deployment
      request.
    validationErrors: Validation errors that would cause the deployment change
      request to be rejected.
  """
    routingChanges = _messages.MessageField('GoogleCloudApigeeV1DeploymentChangeReportRoutingChange', 1, repeated=True)
    routingConflicts = _messages.MessageField('GoogleCloudApigeeV1DeploymentChangeReportRoutingConflict', 2, repeated=True)
    validationErrors = _messages.MessageField('GoogleRpcPreconditionFailure', 3)