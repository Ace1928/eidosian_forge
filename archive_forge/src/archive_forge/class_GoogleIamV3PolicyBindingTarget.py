from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIamV3PolicyBindingTarget(_messages.Message):
    """Target is the full resource name of the resource to which the policy
  will be bound. Immutable once set.

  Fields:
    principalSet: Immutable. Full Resource Name used for principal access
      boundary policy bindings Examples: Organization:
      "//cloudresourcemanager.googleapis.com/organizations/ORGANIZATION_ID"
      Folder: "//cloudresourcemanager.googleapis.com/folders/FOLDER_ID"
      Project: "//cloudresourcemanager.googleapis.com/projects/PROJECT_NUMBER"
      "//cloudresourcemanager.googleapis.com/projects/PROJECT_ID" Workload
      Identity Pool: "//iam.googleapis.com/projects/PROJECT_NUMBER/locations/L
      OCATION/workloadIdentityPools/WORKLOAD_POOL_ID" Workforce Identity:
      "//iam.googleapis.com/locations/global/workforcePools/WORKFORCE_POOL_ID"
      Workspace Identity:
      "//iam.googleapis.com/locations/global/workspace/WORKSPACE_ID"
  """
    principalSet = _messages.StringField(1)