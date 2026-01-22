from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GoogleCloudOrgpolicyV2PolicySpec(_messages.Message):
    """Defines a Google Cloud policy specification which is used to specify
  constraints for configurations of Google Cloud resources.

  Fields:
    etag: An opaque tag indicating the current version of the policySpec, used
      for concurrency control. This field is ignored if used in a
      `CreatePolicy` request. When the policy is returned from either a
      `GetPolicy` or a `ListPolicies` request, this `etag` indicates the
      version of the current policySpec to use when executing a read-modify-
      write loop. When the policy is returned from a `GetEffectivePolicy`
      request, the `etag` will be unset.
    inheritFromParent: Determines the inheritance behavior for this policy. If
      `inherit_from_parent` is true, policy rules set higher up in the
      hierarchy (up to the closest root) are inherited and present in the
      effective policy. If it is false, then no rules are inherited, and this
      policy becomes the new root for evaluation. This field can be set only
      for policies which configure list constraints.
    reset: Ignores policies set above this resource and restores the
      `constraint_default` enforcement behavior of the specific constraint at
      this resource. This field can be set in policies for either list or
      boolean constraints. If set, `rules` must be empty and
      `inherit_from_parent` must be set to false.
    rules: In policies for boolean constraints, the following requirements
      apply: - There must be one and only one policy rule where condition is
      unset. - Boolean policy rules with conditions must set `enforced` to the
      opposite of the policy rule without a condition. - During policy
      evaluation, policy rules with conditions that are true for a target
      resource take precedence.
    updateTime: Output only. The time stamp this was previously updated. This
      represents the last time a call to `CreatePolicy` or `UpdatePolicy` was
      made for that policy.
  """
    etag = _messages.StringField(1)
    inheritFromParent = _messages.BooleanField(2)
    reset = _messages.BooleanField(3)
    rules = _messages.MessageField('GoogleCloudOrgpolicyV2PolicySpecPolicyRule', 4, repeated=True)
    updateTime = _messages.StringField(5)