from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsHydratedDeploymentsGetRequest(_messages.Message):
    """A TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsHydrat
  edDeploymentsGetRequest object.

  Fields:
    name: Required. Name of the hydrated deployment.
  """
    name = _messages.StringField(1, required=True)