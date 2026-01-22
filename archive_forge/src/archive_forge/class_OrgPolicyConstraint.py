from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OrgPolicyConstraint(_messages.Message):
    """Message for Org Policy Canned Constraint.

  Fields:
    cannedConstraintId: Required. Org Policy Canned Constraint id.
    policyRules: Required. Org PolicySpec rules.
  """
    cannedConstraintId = _messages.StringField(1)
    policyRules = _messages.MessageField('GoogleCloudSecuritypostureV1alphaPolicyRule', 2, repeated=True)