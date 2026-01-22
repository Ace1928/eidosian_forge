from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsListRequest(_messages.Message):
    """A
  TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsListRequest
  object.

  Fields:
    filter: Optional. Filtering only supports equality on deployment state. It
      should be in the form: "state = DRAFT". `OR` operator can be used to get
      response for multiple states. e.g. "state = DRAFT OR state = APPLIED".
    pageSize: Optional. The maximum number of deployments to return per page.
    pageToken: Optional. The page token, received from a previous
      ListDeployments call. It can be provided to retrieve the subsequent
      page.
    parent: Required. The name of parent orchestration cluster resource.
      Format should be - "projects/{project_id}/locations/{location_name}/orch
      estrationClusters/{orchestration_cluster}".
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)