from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MembershipConstraint(_messages.Message):
    """Membership specific constraint data.

  Fields:
    constraintRef: The constraint this data refers to.
    content: The string content for the constraint resource.
    kind: The kind of the constraint on this membership, for display purposes.
    membershipRef: The membership this data refers to.
    metadata: Membership-specific constraint metadata.
    spec: Membership-specific constraint spec.
    status: Membership-specific constraint status.
  """
    constraintRef = _messages.MessageField('ConstraintRef', 1)
    content = _messages.MessageField('StringContent', 2)
    kind = _messages.StringField(3)
    membershipRef = _messages.MessageField('MembershipRef', 4)
    metadata = _messages.MessageField('MembershipConstraintMetadata', 5)
    spec = _messages.MessageField('MembershipConstraintSpec', 6)
    status = _messages.MessageField('MembershipConstraintStatus', 7)