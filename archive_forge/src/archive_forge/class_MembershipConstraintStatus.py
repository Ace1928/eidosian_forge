from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MembershipConstraintStatus(_messages.Message):
    """MembershipConstraintStatus contains high-level information from
  constraint status. Omits violation-level information from constraint status,
  which is in separate violation resources.

  Fields:
    auditTimestamp: status.audit_timestamp from the constraint.
    numViolations: status.total_violations from the constraint.
  """
    auditTimestamp = _messages.StringField(1)
    numViolations = _messages.IntegerField(2)