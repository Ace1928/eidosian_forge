from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MembershipConstraintAuditViolation(_messages.Message):
    """MembershipConstraintAuditViolation encodes info relevant to a violation
  of a single constraint on a single member cluster.

  Fields:
    auditTimestamp: The audit timestamp when this violation was observed on
      the membership.
    constraintRef: The constraint ref of the violated constraint.
    errorMessage: An error message detailing the violation.
    membershipRef: The membership this violation occurs on.
    resourceRef: The resource ref of the violating K8S object.
  """
    auditTimestamp = _messages.StringField(1)
    constraintRef = _messages.MessageField('ConstraintRef', 2)
    errorMessage = _messages.StringField(3)
    membershipRef = _messages.MessageField('MembershipRef', 4)
    resourceRef = _messages.MessageField('ResourceRef', 5)