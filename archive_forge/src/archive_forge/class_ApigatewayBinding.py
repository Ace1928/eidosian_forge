from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigatewayBinding(_messages.Message):
    """Associates `members`, or principals, with a `role`.

  Fields:
    condition: The condition that is associated with this binding. If the
      condition evaluates to `true`, then this binding applies to the current
      request. If the condition evaluates to `false`, then this binding does
      not apply to the current request. However, a different role binding
      might grant the same role to one or more of the principals in this
      binding. To learn which resources support conditions in their IAM
      policies, see the [IAM
      documentation](https://cloud.google.com/iam/help/conditions/resource-
      policies).
    members: Specifies the principals requesting access for a Google Cloud
      resource. `members` can have the following values: * `allUsers`: A
      special identifier that represents anyone who is on the internet; with
      or without a Google account. * `allAuthenticatedUsers`: A special
      identifier that represents anyone who is authenticated with a Google
      account or a service account. Does not include identities that come from
      external identity providers (IdPs) through identity federation. *
      `user:{emailid}`: An email address that represents a specific Google
      account. For example, `alice@example.com` . *
      `serviceAccount:{emailid}`: An email address that represents a Google
      service account. For example, `my-other-
      app@appspot.gserviceaccount.com`. *
      `serviceAccount:{projectid}.svc.id.goog[{namespace}/{kubernetes-sa}]`:
      An identifier for a [Kubernetes service
      account](https://cloud.google.com/kubernetes-engine/docs/how-
      to/kubernetes-service-accounts). For example, `my-
      project.svc.id.goog[my-namespace/my-kubernetes-sa]`. *
      `group:{emailid}`: An email address that represents a Google group. For
      example, `admins@example.com`. * `domain:{domain}`: The G Suite domain
      (primary) that represents all the users of that domain. For example,
      `google.com` or `example.com`. * `principal://iam.googleapis.com/locatio
      ns/global/workforcePools/{pool_id}/subject/{subject_attribute_value}`: A
      single identity in a workforce identity pool. * `principalSet://iam.goog
      leapis.com/locations/global/workforcePools/{pool_id}/group/{group_id}`:
      All workforce identities in a group. * `principalSet://iam.googleapis.co
      m/locations/global/workforcePools/{pool_id}/attribute.{attribute_name}/{
      attribute_value}`: All workforce identities with a specific attribute
      value. * `principalSet://iam.googleapis.com/locations/global/workforcePo
      ols/{pool_id}/*`: All identities in a workforce identity pool. * `princi
      pal://iam.googleapis.com/projects/{project_number}/locations/global/work
      loadIdentityPools/{pool_id}/subject/{subject_attribute_value}`: A single
      identity in a workload identity pool. * `principalSet://iam.googleapis.c
      om/projects/{project_number}/locations/global/workloadIdentityPools/{poo
      l_id}/group/{group_id}`: A workload identity pool group. * `principalSet
      ://iam.googleapis.com/projects/{project_number}/locations/global/workloa
      dIdentityPools/{pool_id}/attribute.{attribute_name}/{attribute_value}`:
      All identities in a workload identity pool with a certain attribute. * `
      principalSet://iam.googleapis.com/projects/{project_number}/locations/gl
      obal/workloadIdentityPools/{pool_id}/*`: All identities in a workload
      identity pool. * `deleted:user:{emailid}?uid={uniqueid}`: An email
      address (plus unique identifier) representing a user that has been
      recently deleted. For example,
      `alice@example.com?uid=123456789012345678901`. If the user is recovered,
      this value reverts to `user:{emailid}` and the recovered user retains
      the role in the binding. *
      `deleted:serviceAccount:{emailid}?uid={uniqueid}`: An email address
      (plus unique identifier) representing a service account that has been
      recently deleted. For example, `my-other-
      app@appspot.gserviceaccount.com?uid=123456789012345678901`. If the
      service account is undeleted, this value reverts to
      `serviceAccount:{emailid}` and the undeleted service account retains the
      role in the binding. * `deleted:group:{emailid}?uid={uniqueid}`: An
      email address (plus unique identifier) representing a Google group that
      has been recently deleted. For example,
      `admins@example.com?uid=123456789012345678901`. If the group is
      recovered, this value reverts to `group:{emailid}` and the recovered
      group retains the role in the binding. * `deleted:principal://iam.google
      apis.com/locations/global/workforcePools/{pool_id}/subject/{subject_attr
      ibute_value}`: Deleted single identity in a workforce identity pool. For
      example, `deleted:principal://iam.googleapis.com/locations/global/workfo
      rcePools/my-pool-id/subject/my-subject-attribute-value`.
    role: Role that is assigned to the list of `members`, or principals. For
      example, `roles/viewer`, `roles/editor`, or `roles/owner`. For an
      overview of the IAM roles and permissions, see the [IAM
      documentation](https://cloud.google.com/iam/docs/roles-overview). For a
      list of the available pre-defined roles, see
      [here](https://cloud.google.com/iam/docs/understanding-roles).
  """
    condition = _messages.MessageField('ApigatewayExpr', 1)
    members = _messages.StringField(2, repeated=True)
    role = _messages.StringField(3)