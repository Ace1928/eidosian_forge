from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsHydratedDeploymentsListRequest(_messages.Message):
    """A TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsHydrat
  edDeploymentsListRequest object.

  Fields:
    pageSize: Optional. The maximum number of hydrated deployments to return.
      The service may return fewer than this value. If unspecified, at most 50
      hydrated deployments will be returned. The maximum value is 1000. Values
      above 1000 will be set to 1000.
    pageToken: Optional. The page token, received from a previous
      ListHydratedDeployments call. Provide this to retrieve the subsequent
      page.
    parent: Required. The deployment managing the hydrated deployments.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)