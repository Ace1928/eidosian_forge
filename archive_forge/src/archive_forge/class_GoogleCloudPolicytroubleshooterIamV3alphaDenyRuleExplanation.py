from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicytroubleshooterIamV3alphaDenyRuleExplanation(_messages.Message):
    """Details about how a deny rule in a deny policy affects a principal's
  ability to use a permission.

  Enums:
    DenyAccessStateValueValuesEnum: Required. Indicates whether _this rule_
      denies the specified permission to the specified principal for the
      specified resource. This field does _not_ indicate whether the principal
      is actually denied on the permission for the resource. There might be
      another rule that overrides this rule. To determine whether the
      principal actually has the permission, use the `overall_access_state`
      field in the TroubleshootIamPolicyResponse.
    RelevanceValueValuesEnum: The relevance of this role binding to the
      overall determination for the entire policy.

  Messages:
    DeniedPermissionsValue: Lists all denied permissions in the deny rule and
      indicates whether each permission matches the permission in the request.
      Each key identifies a denied permission in the rule, and each value
      indicates whether the denied permission matches the permission in the
      request.
    DeniedPrincipalsValue: Lists all denied principals in the deny rule and
      indicates whether each principal matches the principal in the request,
      either directly or through membership in a principal set. Each key
      identifies a denied principal in the rule, and each value indicates
      whether the denied principal matches the principal in the request.
    ExceptionPermissionsValue: Lists all exception permissions in the deny
      rule and indicates whether each permission matches the permission in the
      request. Each key identifies a exception permission in the rule, and
      each value indicates whether the exception permission matches the
      permission in the request.
    ExceptionPrincipalsValue: Lists all exception principals in the deny rule
      and indicates whether each principal matches the principal in the
      request, either directly or through membership in a principal set. Each
      key identifies a exception principal in the rule, and each value
      indicates whether the exception principal matches the principal in the
      request.

  Fields:
    combinedDeniedPermission: Indicates whether the permission in the request
      is listed as a denied permission in the deny rule.
    combinedDeniedPrincipal: Indicates whether the principal is listed as a
      denied principal in the deny rule, either directly or through membership
      in a principal set.
    combinedExceptionPermission: Indicates whether the permission in the
      request is listed as an exception permission in the deny rule.
    combinedExceptionPrincipal: Indicates whether the principal is listed as
      an exception principal in the deny rule, either directly or through
      membership in a principal set.
    condition: A condition expression that specifies when the deny rule denies
      the principal access. To learn about IAM Conditions, see
      https://cloud.google.com/iam/help/conditions/overview.
    conditionExplanation: Condition evaluation state for this role binding.
    deniedPermissions: Lists all denied permissions in the deny rule and
      indicates whether each permission matches the permission in the request.
      Each key identifies a denied permission in the rule, and each value
      indicates whether the denied permission matches the permission in the
      request.
    deniedPrincipals: Lists all denied principals in the deny rule and
      indicates whether each principal matches the principal in the request,
      either directly or through membership in a principal set. Each key
      identifies a denied principal in the rule, and each value indicates
      whether the denied principal matches the principal in the request.
    denyAccessState: Required. Indicates whether _this rule_ denies the
      specified permission to the specified principal for the specified
      resource. This field does _not_ indicate whether the principal is
      actually denied on the permission for the resource. There might be
      another rule that overrides this rule. To determine whether the
      principal actually has the permission, use the `overall_access_state`
      field in the TroubleshootIamPolicyResponse.
    exceptionPermissions: Lists all exception permissions in the deny rule and
      indicates whether each permission matches the permission in the request.
      Each key identifies a exception permission in the rule, and each value
      indicates whether the exception permission matches the permission in the
      request.
    exceptionPrincipals: Lists all exception principals in the deny rule and
      indicates whether each principal matches the principal in the request,
      either directly or through membership in a principal set. Each key
      identifies a exception principal in the rule, and each value indicates
      whether the exception principal matches the principal in the request.
    relevance: The relevance of this role binding to the overall determination
      for the entire policy.
  """

    class DenyAccessStateValueValuesEnum(_messages.Enum):
        """Required. Indicates whether _this rule_ denies the specified
    permission to the specified principal for the specified resource. This
    field does _not_ indicate whether the principal is actually denied on the
    permission for the resource. There might be another rule that overrides
    this rule. To determine whether the principal actually has the permission,
    use the `overall_access_state` field in the TroubleshootIamPolicyResponse.

    Values:
      DENY_ACCESS_STATE_UNSPECIFIED: Not specified.
      DENY_ACCESS_STATE_DENIED: The deny policy denies the principal the
        permission.
      DENY_ACCESS_STATE_NOT_DENIED: The deny policy doesn't deny the principal
        the permission.
      DENY_ACCESS_STATE_UNKNOWN_CONDITIONAL: The deny policy denies the
        principal the permission if a condition expression evaluates to
        `true`. However, the sender of the request didn't provide enough
        context for Policy Troubleshooter to evaluate the condition
        expression.
      DENY_ACCESS_STATE_UNKNOWN_INFO: The sender of the request does not have
        access to all of the deny policies that Policy Troubleshooter needs to
        evaluate the principal's access.
    """
        DENY_ACCESS_STATE_UNSPECIFIED = 0
        DENY_ACCESS_STATE_DENIED = 1
        DENY_ACCESS_STATE_NOT_DENIED = 2
        DENY_ACCESS_STATE_UNKNOWN_CONDITIONAL = 3
        DENY_ACCESS_STATE_UNKNOWN_INFO = 4

    class RelevanceValueValuesEnum(_messages.Enum):
        """The relevance of this role binding to the overall determination for
    the entire policy.

    Values:
      HEURISTIC_RELEVANCE_UNSPECIFIED: Not specified.
      HEURISTIC_RELEVANCE_NORMAL: The data point has a limited effect on the
        result. Changing the data point is unlikely to affect the overall
        determination.
      HEURISTIC_RELEVANCE_HIGH: The data point has a strong effect on the
        result. Changing the data point is likely to affect the overall
        determination.
    """
        HEURISTIC_RELEVANCE_UNSPECIFIED = 0
        HEURISTIC_RELEVANCE_NORMAL = 1
        HEURISTIC_RELEVANCE_HIGH = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class DeniedPermissionsValue(_messages.Message):
        """Lists all denied permissions in the deny rule and indicates whether
    each permission matches the permission in the request. Each key identifies
    a denied permission in the rule, and each value indicates whether the
    denied permission matches the permission in the request.

    Messages:
      AdditionalProperty: An additional property for a DeniedPermissionsValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        DeniedPermissionsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a DeniedPermissionsValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudPolicytroubleshooterIamV3alphaDenyRuleExplanationA
          nnotatedPermissionMatching attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('GoogleCloudPolicytroubleshooterIamV3alphaDenyRuleExplanationAnnotatedPermissionMatching', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class DeniedPrincipalsValue(_messages.Message):
        """Lists all denied principals in the deny rule and indicates whether
    each principal matches the principal in the request, either directly or
    through membership in a principal set. Each key identifies a denied
    principal in the rule, and each value indicates whether the denied
    principal matches the principal in the request.

    Messages:
      AdditionalProperty: An additional property for a DeniedPrincipalsValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        DeniedPrincipalsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a DeniedPrincipalsValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudPolicytroubleshooterIamV3alphaDenyRuleExplanationA
          nnotatedDenyPrincipalMatching attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('GoogleCloudPolicytroubleshooterIamV3alphaDenyRuleExplanationAnnotatedDenyPrincipalMatching', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ExceptionPermissionsValue(_messages.Message):
        """Lists all exception permissions in the deny rule and indicates whether
    each permission matches the permission in the request. Each key identifies
    a exception permission in the rule, and each value indicates whether the
    exception permission matches the permission in the request.

    Messages:
      AdditionalProperty: An additional property for a
        ExceptionPermissionsValue object.

    Fields:
      additionalProperties: Additional properties of type
        ExceptionPermissionsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ExceptionPermissionsValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudPolicytroubleshooterIamV3alphaDenyRuleExplanationA
          nnotatedPermissionMatching attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('GoogleCloudPolicytroubleshooterIamV3alphaDenyRuleExplanationAnnotatedPermissionMatching', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ExceptionPrincipalsValue(_messages.Message):
        """Lists all exception principals in the deny rule and indicates whether
    each principal matches the principal in the request, either directly or
    through membership in a principal set. Each key identifies a exception
    principal in the rule, and each value indicates whether the exception
    principal matches the principal in the request.

    Messages:
      AdditionalProperty: An additional property for a
        ExceptionPrincipalsValue object.

    Fields:
      additionalProperties: Additional properties of type
        ExceptionPrincipalsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ExceptionPrincipalsValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudPolicytroubleshooterIamV3alphaDenyRuleExplanationA
          nnotatedDenyPrincipalMatching attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('GoogleCloudPolicytroubleshooterIamV3alphaDenyRuleExplanationAnnotatedDenyPrincipalMatching', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    combinedDeniedPermission = _messages.MessageField('GoogleCloudPolicytroubleshooterIamV3alphaDenyRuleExplanationAnnotatedPermissionMatching', 1)
    combinedDeniedPrincipal = _messages.MessageField('GoogleCloudPolicytroubleshooterIamV3alphaDenyRuleExplanationAnnotatedDenyPrincipalMatching', 2)
    combinedExceptionPermission = _messages.MessageField('GoogleCloudPolicytroubleshooterIamV3alphaDenyRuleExplanationAnnotatedPermissionMatching', 3)
    combinedExceptionPrincipal = _messages.MessageField('GoogleCloudPolicytroubleshooterIamV3alphaDenyRuleExplanationAnnotatedDenyPrincipalMatching', 4)
    condition = _messages.MessageField('GoogleTypeExpr', 5)
    conditionExplanation = _messages.MessageField('GoogleCloudPolicytroubleshooterIamV3alphaConditionExplanation', 6)
    deniedPermissions = _messages.MessageField('DeniedPermissionsValue', 7)
    deniedPrincipals = _messages.MessageField('DeniedPrincipalsValue', 8)
    denyAccessState = _messages.EnumField('DenyAccessStateValueValuesEnum', 9)
    exceptionPermissions = _messages.MessageField('ExceptionPermissionsValue', 10)
    exceptionPrincipals = _messages.MessageField('ExceptionPrincipalsValue', 11)
    relevance = _messages.EnumField('RelevanceValueValuesEnum', 12)