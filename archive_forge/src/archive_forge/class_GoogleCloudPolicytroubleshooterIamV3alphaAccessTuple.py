from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicytroubleshooterIamV3alphaAccessTuple(_messages.Message):
    """Information about the principal, resource, and permission to check.

  Fields:
    conditionContext: Optional. Additional context for the request, such as
      the request time or IP address. This context allows Policy
      Troubleshooter to troubleshoot conditional role bindings and deny rules.
    fullResourceName: Required. The full resource name that identifies the
      resource. For example, `//compute.googleapis.com/projects/my-
      project/zones/us-central1-a/instances/my-instance`. For examples of full
      resource names for Google Cloud services, see
      https://cloud.google.com/iam/help/troubleshooter/full-resource-names.
    permission: Required. The IAM permission to check for, either in the `v1`
      permission format or the `v2` permission format. For a complete list of
      IAM permissions in the `v1` format, see
      https://cloud.google.com/iam/help/permissions/reference. For a list of
      IAM permissions in the `v2` format, see
      https://cloud.google.com/iam/help/deny/supported-permissions. For a
      complete list of predefined IAM roles and the permissions in each role,
      see https://cloud.google.com/iam/help/roles/reference.
    permissionFqdn: Output only. The permission that Policy Troubleshooter
      checked for, in the `v2` format.
    principal: Required. The email address of the principal whose access you
      want to check. For example, `alice@example.com` or `my-service-
      account@my-project.iam.gserviceaccount.com`. The principal must be a
      Google Account or a service account. Other types of principals are not
      supported.
  """
    conditionContext = _messages.MessageField('GoogleCloudPolicytroubleshooterIamV3alphaConditionContext', 1)
    fullResourceName = _messages.StringField(2)
    permission = _messages.StringField(3)
    permissionFqdn = _messages.StringField(4)
    principal = _messages.StringField(5)