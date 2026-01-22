from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AssuredworkloadsOrganizationsLocationsWorkloadsPatchRequest(_messages.Message):
    """A AssuredworkloadsOrganizationsLocationsWorkloadsPatchRequest object.

  Fields:
    googleCloudAssuredworkloadsV1Workload: A
      GoogleCloudAssuredworkloadsV1Workload resource to be passed as the
      request body.
    name: Optional. The resource name of the workload. Format:
      organizations/{organization}/locations/{location}/workloads/{workload}
      Read-only.
    updateMask: Required. The list of fields to be updated.
  """
    googleCloudAssuredworkloadsV1Workload = _messages.MessageField('GoogleCloudAssuredworkloadsV1Workload', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)