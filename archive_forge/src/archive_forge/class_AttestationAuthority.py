from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AttestationAuthority(_messages.Message):
    """Note kind that represents a logical attestation "role" or "authority".
  For example, an organization might have one `AttestationAuthority` for "QA"
  and one for "build". This Note is intended to act strictly as a grouping
  mechanism for the attached Occurrences (Attestations). This grouping
  mechanism also provides a security boundary, since IAM ACLs gate the ability
  for a principle to attach an Occurrence to a given Note. It also provides a
  single point of lookup to find all attached Attestation Occurrences, even if
  they don't all live in the same project.

  Fields:
    hint: A AttestationAuthorityHint attribute.
  """
    hint = _messages.MessageField('AttestationAuthorityHint', 1)