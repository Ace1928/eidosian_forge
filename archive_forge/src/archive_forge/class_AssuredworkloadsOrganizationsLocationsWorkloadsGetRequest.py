from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AssuredworkloadsOrganizationsLocationsWorkloadsGetRequest(_messages.Message):
    """A AssuredworkloadsOrganizationsLocationsWorkloadsGetRequest object.

  Fields:
    name: Required. The resource name of the Workload to fetch. This is the
      workloads's relative path in the API, formatted as "organizations/{organ
      ization_id}/locations/{location_id}/workloads/{workload_id}". For
      example, "organizations/123/locations/us-east1/workloads/assured-
      workload-1".
  """
    name = _messages.StringField(1, required=True)