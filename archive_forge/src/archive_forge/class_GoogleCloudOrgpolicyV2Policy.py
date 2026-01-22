from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GoogleCloudOrgpolicyV2Policy(_messages.Message):
    """Defines an organization policy which is used to specify constraints for
  configurations of Google Cloud resources.

  Fields:
    alternate: Deprecated.
    dryRunSpec: Dry-run policy. Audit-only policy, can be used to monitor how
      the policy would have impacted the existing and future resources if it's
      enforced.
    etag: Optional. An opaque tag indicating the current state of the policy,
      used for concurrency control. This 'etag' is computed by the server
      based on the value of other fields, and may be sent on update and delete
      requests to ensure the client has an up-to-date value before proceeding.
    name: Immutable. The resource name of the policy. Must be one of the
      following forms, where `constraint_name` is the name of the constraint
      which this policy configures: *
      `projects/{project_number}/policies/{constraint_name}` *
      `folders/{folder_id}/policies/{constraint_name}` *
      `organizations/{organization_id}/policies/{constraint_name}` For
      example, `projects/123/policies/compute.disableSerialPortAccess`. Note:
      `projects/{project_id}/policies/{constraint_name}` is also an acceptable
      name for API requests, but responses will return the name using the
      equivalent project number.
    spec: Basic information about the Organization Policy.
  """
    alternate = _messages.MessageField('GoogleCloudOrgpolicyV2AlternatePolicySpec', 1)
    dryRunSpec = _messages.MessageField('GoogleCloudOrgpolicyV2PolicySpec', 2)
    etag = _messages.StringField(3)
    name = _messages.StringField(4)
    spec = _messages.MessageField('GoogleCloudOrgpolicyV2PolicySpec', 5)