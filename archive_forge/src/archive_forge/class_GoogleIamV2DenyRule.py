from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIamV2DenyRule(_messages.Message):
    """A deny rule in an IAM deny policy.

  Fields:
    denialCondition: The condition that determines whether this deny rule
      applies to a request. If the condition expression evaluates to `true`,
      then the deny rule is applied; otherwise, the deny rule is not applied.
      Each deny rule is evaluated independently. If this deny rule does not
      apply to a request, other deny rules might still apply. The condition
      can use CEL functions that evaluate [resource
      tags](https://cloud.google.com/iam/help/conditions/resource-tags). Other
      functions and operators are not supported.
    deniedPermissions: The permissions that are explicitly denied by this
      rule. Each permission uses the format
      `{service_fqdn}/{resource}.{verb}`, where `{service_fqdn}` is the fully
      qualified domain name for the service. For example,
      `iam.googleapis.com/roles.list`.
    deniedPrincipals: The identities that are prevented from using one or more
      permissions on Google Cloud resources. This field can contain the
      following values: * `principal://goog/subject/{email_id}`: A specific
      Google Account. Includes Gmail, Cloud Identity, and Google Workspace
      user accounts. For example,
      `principal://goog/subject/alice@example.com`. * `principal://iam.googlea
      pis.com/projects/-/serviceAccounts/{service_account_id}`: A Google Cloud
      service account. For example,
      `principal://iam.googleapis.com/projects/-/serviceAccounts/my-service-
      account@iam.gserviceaccount.com`. *
      `principalSet://goog/group/{group_id}`: A Google group. For example,
      `principalSet://goog/group/admins@example.com`. *
      `principalSet://goog/public:all`: A special identifier that represents
      any principal that is on the internet, even if they do not have a Google
      Account or are not logged in. *
      `principalSet://goog/cloudIdentityCustomerId/{customer_id}`: All of the
      principals associated with the specified Google Workspace or Cloud
      Identity customer ID. For example,
      `principalSet://goog/cloudIdentityCustomerId/C01Abc35`. * `principal://i
      am.googleapis.com/locations/global/workforcePools/{pool_id}/subject/{sub
      ject_attribute_value}`: A single identity in a workforce identity pool.
      * `principalSet://iam.googleapis.com/locations/global/workforcePools/{po
      ol_id}/group/{group_id}`: All workforce identities in a group. * `princi
      palSet://iam.googleapis.com/locations/global/workforcePools/{pool_id}/at
      tribute.{attribute_name}/{attribute_value}`: All workforce identities
      with a specific attribute value. * `principalSet://iam.googleapis.com/lo
      cations/global/workforcePools/{pool_id}/*`: All identities in a
      workforce identity pool. * `principal://iam.googleapis.com/projects/{pro
      ject_number}/locations/global/workloadIdentityPools/{pool_id}/subject/{s
      ubject_attribute_value}`: A single identity in a workload identity pool.
      * `principalSet://iam.googleapis.com/projects/{project_number}/locations
      /global/workloadIdentityPools/{pool_id}/group/{group_id}`: A workload
      identity pool group. * `principalSet://iam.googleapis.com/projects/{proj
      ect_number}/locations/global/workloadIdentityPools/{pool_id}/attribute.{
      attribute_name}/{attribute_value}`: All identities in a workload
      identity pool with a certain attribute. * `principalSet://iam.googleapis
      .com/projects/{project_number}/locations/global/workloadIdentityPools/{p
      ool_id}/*`: All identities in a workload identity pool. *
      `deleted:principal://goog/subject/{email_id}?uid={uid}`: A specific
      Google Account that was deleted recently. For example,
      `deleted:principal://goog/subject/alice@example.com?uid=1234567890`. If
      the Google Account is recovered, this identifier reverts to the standard
      identifier for a Google Account. *
      `deleted:principalSet://goog/group/{group_id}?uid={uid}`: A Google group
      that was deleted recently. For example,
      `deleted:principalSet://goog/group/admins@example.com?uid=1234567890`.
      If the Google group is restored, this identifier reverts to the standard
      identifier for a Google group. * `deleted:principal://iam.googleapis.com
      /projects/-/serviceAccounts/{service_account_id}?uid={uid}`: A Google
      Cloud service account that was deleted recently. For example,
      `deleted:principal://iam.googleapis.com/projects/-/serviceAccounts/my-
      service-account@iam.gserviceaccount.com?uid=1234567890`. If the service
      account is undeleted, this identifier reverts to the standard identifier
      for a service account. * `deleted:principal://iam.googleapis.com/locatio
      ns/global/workforcePools/{pool_id}/subject/{subject_attribute_value}`:
      Deleted single identity in a workforce identity pool. For example, `dele
      ted:principal://iam.googleapis.com/locations/global/workforcePools/my-
      pool-id/subject/my-subject-attribute-value`.
    exceptionPermissions: Specifies the permissions that this rule excludes
      from the set of denied permissions given by `denied_permissions`. If a
      permission appears in `denied_permissions` _and_ in
      `exception_permissions` then it will _not_ be denied. The excluded
      permissions can be specified using the same syntax as
      `denied_permissions`.
    exceptionPrincipals: The identities that are excluded from the deny rule,
      even if they are listed in the `denied_principals`. For example, you
      could add a Google group to the `denied_principals`, then exclude
      specific users who belong to that group. This field can contain the same
      values as the `denied_principals` field, excluding
      `principalSet://goog/public:all`, which represents all users on the
      internet.
  """
    denialCondition = _messages.MessageField('Expr', 1)
    deniedPermissions = _messages.StringField(2, repeated=True)
    deniedPrincipals = _messages.StringField(3, repeated=True)
    exceptionPermissions = _messages.StringField(4, repeated=True)
    exceptionPrincipals = _messages.StringField(5, repeated=True)