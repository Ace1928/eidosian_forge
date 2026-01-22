from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicysimulatorV1betaBindingExplanation(_messages.Message):
    """Details about how a binding in a policy affects a principal's ability to
  use a permission.

  Enums:
    AccessValueValuesEnum: Required. Indicates whether _this binding_ provides
      the specified permission to the specified principal for the specified
      resource. This field does _not_ indicate whether the principal actually
      has the permission for the resource. There might be another binding that
      overrides this binding. To determine whether the principal actually has
      the permission, use the `access` field in the
      TroubleshootIamPolicyResponse.
    RelevanceValueValuesEnum: The relevance of this binding to the overall
      determination for the entire policy.
    RolePermissionValueValuesEnum: Indicates whether the role granted by this
      binding contains the specified permission.
    RolePermissionRelevanceValueValuesEnum: The relevance of the permission's
      existence, or nonexistence, in the role to the overall determination for
      the entire policy.

  Messages:
    MembershipsValue: Indicates whether each principal in the binding includes
      the principal specified in the request, either directly or indirectly.
      Each key identifies a principal in the binding, and each value indicates
      whether the principal in the binding includes the principal in the
      request. For example, suppose that a binding includes the following
      principals: * `user:alice@example.com` * `group:product-eng@example.com`
      The principal in the replayed access tuple is `user:bob@example.com`.
      This user is a principal of the group `group:product-eng@example.com`.
      For the first principal in the binding, the key is
      `user:alice@example.com`, and the `membership` field in the value is set
      to `MEMBERSHIP_NOT_INCLUDED`. For the second principal in the binding,
      the key is `group:product-eng@example.com`, and the `membership` field
      in the value is set to `MEMBERSHIP_INCLUDED`.

  Fields:
    access: Required. Indicates whether _this binding_ provides the specified
      permission to the specified principal for the specified resource. This
      field does _not_ indicate whether the principal actually has the
      permission for the resource. There might be another binding that
      overrides this binding. To determine whether the principal actually has
      the permission, use the `access` field in the
      TroubleshootIamPolicyResponse.
    condition: A condition expression that prevents this binding from granting
      access unless the expression evaluates to `true`. To learn about IAM
      Conditions, see https://cloud.google.com/iam/docs/conditions-overview.
    memberships: Indicates whether each principal in the binding includes the
      principal specified in the request, either directly or indirectly. Each
      key identifies a principal in the binding, and each value indicates
      whether the principal in the binding includes the principal in the
      request. For example, suppose that a binding includes the following
      principals: * `user:alice@example.com` * `group:product-eng@example.com`
      The principal in the replayed access tuple is `user:bob@example.com`.
      This user is a principal of the group `group:product-eng@example.com`.
      For the first principal in the binding, the key is
      `user:alice@example.com`, and the `membership` field in the value is set
      to `MEMBERSHIP_NOT_INCLUDED`. For the second principal in the binding,
      the key is `group:product-eng@example.com`, and the `membership` field
      in the value is set to `MEMBERSHIP_INCLUDED`.
    relevance: The relevance of this binding to the overall determination for
      the entire policy.
    role: The role that this binding grants. For example,
      `roles/compute.serviceAgent`. For a complete list of predefined IAM
      roles, as well as the permissions in each role, see
      https://cloud.google.com/iam/help/roles/reference.
    rolePermission: Indicates whether the role granted by this binding
      contains the specified permission.
    rolePermissionRelevance: The relevance of the permission's existence, or
      nonexistence, in the role to the overall determination for the entire
      policy.
  """

    class AccessValueValuesEnum(_messages.Enum):
        """Required. Indicates whether _this binding_ provides the specified
    permission to the specified principal for the specified resource. This
    field does _not_ indicate whether the principal actually has the
    permission for the resource. There might be another binding that overrides
    this binding. To determine whether the principal actually has the
    permission, use the `access` field in the TroubleshootIamPolicyResponse.

    Values:
      ACCESS_STATE_UNSPECIFIED: Default value. This value is unused.
      GRANTED: The principal has the permission.
      NOT_GRANTED: The principal does not have the permission.
      UNKNOWN_CONDITIONAL: The principal has the permission only if a
        condition expression evaluates to `true`.
      UNKNOWN_INFO_DENIED: The user who created the Replay does not have
        access to all of the policies that Policy Simulator needs to evaluate.
    """
        ACCESS_STATE_UNSPECIFIED = 0
        GRANTED = 1
        NOT_GRANTED = 2
        UNKNOWN_CONDITIONAL = 3
        UNKNOWN_INFO_DENIED = 4

    class RelevanceValueValuesEnum(_messages.Enum):
        """The relevance of this binding to the overall determination for the
    entire policy.

    Values:
      HEURISTIC_RELEVANCE_UNSPECIFIED: Default value. This value is unused.
      NORMAL: The data point has a limited effect on the result. Changing the
        data point is unlikely to affect the overall determination.
      HIGH: The data point has a strong effect on the result. Changing the
        data point is likely to affect the overall determination.
    """
        HEURISTIC_RELEVANCE_UNSPECIFIED = 0
        NORMAL = 1
        HIGH = 2

    class RolePermissionRelevanceValueValuesEnum(_messages.Enum):
        """The relevance of the permission's existence, or nonexistence, in the
    role to the overall determination for the entire policy.

    Values:
      HEURISTIC_RELEVANCE_UNSPECIFIED: Default value. This value is unused.
      NORMAL: The data point has a limited effect on the result. Changing the
        data point is unlikely to affect the overall determination.
      HIGH: The data point has a strong effect on the result. Changing the
        data point is likely to affect the overall determination.
    """
        HEURISTIC_RELEVANCE_UNSPECIFIED = 0
        NORMAL = 1
        HIGH = 2

    class RolePermissionValueValuesEnum(_messages.Enum):
        """Indicates whether the role granted by this binding contains the
    specified permission.

    Values:
      ROLE_PERMISSION_UNSPECIFIED: Default value. This value is unused.
      ROLE_PERMISSION_INCLUDED: The permission is included in the role.
      ROLE_PERMISSION_NOT_INCLUDED: The permission is not included in the
        role.
      ROLE_PERMISSION_UNKNOWN_INFO_DENIED: The user who created the Replay is
        not allowed to access the binding.
    """
        ROLE_PERMISSION_UNSPECIFIED = 0
        ROLE_PERMISSION_INCLUDED = 1
        ROLE_PERMISSION_NOT_INCLUDED = 2
        ROLE_PERMISSION_UNKNOWN_INFO_DENIED = 3

    @encoding.MapUnrecognizedFields('additionalProperties')
    class MembershipsValue(_messages.Message):
        """Indicates whether each principal in the binding includes the principal
    specified in the request, either directly or indirectly. Each key
    identifies a principal in the binding, and each value indicates whether
    the principal in the binding includes the principal in the request. For
    example, suppose that a binding includes the following principals: *
    `user:alice@example.com` * `group:product-eng@example.com` The principal
    in the replayed access tuple is `user:bob@example.com`. This user is a
    principal of the group `group:product-eng@example.com`. For the first
    principal in the binding, the key is `user:alice@example.com`, and the
    `membership` field in the value is set to `MEMBERSHIP_NOT_INCLUDED`. For
    the second principal in the binding, the key is `group:product-
    eng@example.com`, and the `membership` field in the value is set to
    `MEMBERSHIP_INCLUDED`.

    Messages:
      AdditionalProperty: An additional property for a MembershipsValue
        object.

    Fields:
      additionalProperties: Additional properties of type MembershipsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a MembershipsValue object.

      Fields:
        key: Name of the additional property.
        value: A
          GoogleCloudPolicysimulatorV1betaBindingExplanationAnnotatedMembershi
          p attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('GoogleCloudPolicysimulatorV1betaBindingExplanationAnnotatedMembership', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    access = _messages.EnumField('AccessValueValuesEnum', 1)
    condition = _messages.MessageField('GoogleTypeExpr', 2)
    memberships = _messages.MessageField('MembershipsValue', 3)
    relevance = _messages.EnumField('RelevanceValueValuesEnum', 4)
    role = _messages.StringField(5)
    rolePermission = _messages.EnumField('RolePermissionValueValuesEnum', 6)
    rolePermissionRelevance = _messages.EnumField('RolePermissionRelevanceValueValuesEnum', 7)