from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamProjectsLocationsWorkloadIdentityPoolsNamespacesManagedIdentitiesWorkloadSourcesPatchRequest(_messages.Message):
    """A IamProjectsLocationsWorkloadIdentityPoolsNamespacesManagedIdentitiesWo
  rkloadSourcesPatchRequest object.

  Fields:
    name: Output only. The resource name of the workload source. If ID of the
      WorkloadSource resource determines which workloads may be matched. The
      following formats are supported: - `project-{project_number}` matches
      workloads within the referenced Google Cloud project.
    updateMask: Required. The list of fields to update.
    workloadSource: A WorkloadSource resource to be passed as the request
      body.
  """
    name = _messages.StringField(1, required=True)
    updateMask = _messages.StringField(2)
    workloadSource = _messages.MessageField('WorkloadSource', 3)