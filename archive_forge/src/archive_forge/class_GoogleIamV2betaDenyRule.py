from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIamV2betaDenyRule(_messages.Message):
    """IAM Deny Policy Rule.

  Fields:
    denialCondition: The condition that is associated with this deny rule.
      NOTE: A satisfied condition will explicitly deny access for applicable
      principal, permission, and resource. Different deny rules, including
      their conditions, are examined independently. Only tag based conditions
      are supported.
    deniedPermissions: Specifies the permissions that are explicitly denied by
      this rule. The denied permission can be specified in the following ways:
      * a full permission string in the following canonical formats as
      described in the service specific documentation: --
      `{service_FQDN}/{resource}.{verb}`. For example,
      `iam.googleapis.com/roles.list`.
    deniedPrincipals: A string attribute.
    exceptionPermissions: Specifies the permissions that are excluded from the
      set of denied permissions given by `denied_permissions`. If a permission
      appears in `denied_permissions` _and_ in `excluded_permissions` then it
      will _not_ be denied. The excluded permissions can be specified using
      the same syntax as `denied_permissions`.
    exceptionPrincipals: Specifies the identities requesting access for a
      Cloud Platform resource, which are excluded from the deny rule.
      `exception_principals` can have the following values: * Google and G
      Suite user accounts: -- `principal://goog/subject/{emailId}`: An email
      address that represents a specific Google account. For example,
      `principal://goog/subject/alice@gmail.com`. * Google and G Suite groups:
      -- `principalSet://goog/group/{groupId}`: An identifier that represents
      a Google group. For example,
      `principalSet://goog/group/admins@example.com`. * Service Accounts: -- `
      principal://iam.googleapis.com/projects/-/serviceAccounts/{serviceAccoun
      tId}`: An identifier that represents a service account. For example, `pr
      incipal://iam.googleapis.com/projects/-/serviceAccounts/sa123@iam.gservi
      ceaccount.com`. * G Suite Customers: --
      `principalSet://goog/cloudIdentityCustomerId/{customerId}`: All of the
      principals associated with the specified G Suite Customer ID. For
      example, `principalSet://goog/cloudIdentityCustomerId/C01Abc35`.
  """
    denialCondition = _messages.MessageField('GoogleTypeExpr', 1)
    deniedPermissions = _messages.StringField(2, repeated=True)
    deniedPrincipals = _messages.StringField(3, repeated=True)
    exceptionPermissions = _messages.StringField(4, repeated=True)
    exceptionPrincipals = _messages.StringField(5, repeated=True)