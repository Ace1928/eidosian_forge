from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRunV1Condition(_messages.Message):
    """Conditions show the status of reconciliation progress on a given
  resource. Most resource use a top-level condition type "Ready" or
  "Completed" to show overall status with other conditions to checkpoint each
  stage of reconciliation. Note that if metadata.Generation does not equal
  status.ObservedGeneration, the conditions shown may not be relevant for the
  current spec.

  Fields:
    lastTransitionTime: Optional. Last time the condition transitioned from
      one status to another.
    message: Optional. Human readable message indicating details about the
      current status.
    reason: Optional. One-word CamelCase reason for the condition's last
      transition. These are intended to be stable, unique values which the
      client may use to trigger error handling logic, whereas messages which
      may be changed later by the server.
    severity: Optional. How to interpret this condition. One of Error,
      Warning, or Info. Conditions of severity Info do not contribute to
      resource readiness.
    status: Status of the condition, one of True, False, Unknown.
    type: type is used to communicate the status of the reconciliation
      process. Types common to all resources include: * "Ready" or
      "Completed": True when the Resource is ready.
  """
    lastTransitionTime = _messages.StringField(1)
    message = _messages.StringField(2)
    reason = _messages.StringField(3)
    severity = _messages.StringField(4)
    status = _messages.StringField(5)
    type = _messages.StringField(6)