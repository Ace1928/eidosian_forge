from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AssuredworkloadsOrganizationsLocationsWorkloadsEnableResourceMonitoringRequest(_messages.Message):
    """A AssuredworkloadsOrganizationsLocationsWorkloadsEnableResourceMonitorin
  gRequest object.

  Fields:
    name: Required. The `name` field is used to identify the workload. Format:
      organizations/{org_id}/locations/{location_id}/workloads/{workload_id}
  """
    name = _messages.StringField(1, required=True)