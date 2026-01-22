from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritypostureV1alphaPolicyRule(_messages.Message):
    """A rule used to express this policy.

  Fields:
    allowAll: Setting this to true means that all values are allowed. This
      field can be set only in policies for list constraints.
    condition: A condition which determines whether this rule is used in the
      evaluation of the policy. When set, the `expression` field in the `Expr'
      must include from 1 to 10 subexpressions, joined by the "||" or "&&"
      operators. Each subexpression must be of the form
      "resource.matchTag('/tag_key_short_name, 'tag_value_short_name')". or
      "resource.matchTagId('tagKeys/key_id', 'tagValues/value_id')". where
      key_name and value_name are the resource names for Label Keys and
      Values. These names are available from the Tag Manager Service. An
      example expression is: "resource.matchTag('123456789/environment,
      'prod')". or "resource.matchTagId('tagKeys/123', 'tagValues/456')".
    denyAll: Setting this to true means that all values are denied. This field
      can be set only in policies for list constraints.
    enforce: If `true`, then the policy is enforced. If `false`, then any
      configuration is acceptable. This field can be set only in policies for
      boolean constraints.
    values: List of values to be used for this policy rule. This field can be
      set only in policies for list constraints.
  """
    allowAll = _messages.BooleanField(1)
    condition = _messages.MessageField('Expr', 2)
    denyAll = _messages.BooleanField(3)
    enforce = _messages.BooleanField(4)
    values = _messages.MessageField('GoogleCloudSecuritypostureV1alphaPolicyRuleStringValues', 5)