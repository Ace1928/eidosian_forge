from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RestrictionEvaluations(_messages.Message):
    """Evaluations of restrictions applied to parent group on this membership.

  Fields:
    memberRestrictionEvaluation: Evaluation of the member restriction applied
      to this membership. Empty if the user lacks permission to view the
      restriction evaluation.
  """
    memberRestrictionEvaluation = _messages.MessageField('MembershipRoleRestrictionEvaluation', 1)