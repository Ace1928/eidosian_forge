from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV2PolicyDriftDetails(_messages.Message):
    """The policy field that violates the deployed posture and its expected and
  detected values.

  Fields:
    detectedValue: The detected value that violates the deployed posture, for
      example, `false` or `allowed_values={"projects/22831892"}`.
    expectedValue: The value of this field that was configured in a posture,
      for example, `true` or `allowed_values={"projects/29831892"}`.
    field: The name of the updated field, for example
      constraint.implementation.policy_rules[0].enforce
  """
    detectedValue = _messages.StringField(1)
    expectedValue = _messages.StringField(2)
    field = _messages.StringField(3)