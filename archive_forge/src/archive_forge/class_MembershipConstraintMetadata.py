from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MembershipConstraintMetadata(_messages.Message):
    """MembershipConstraintMetadata contains relevant fields from constraint
  metadata.

  Fields:
    constraintInfo: constraint bundle information from the metadata.annotation
      field.
    creation: metadata.creation_timestamp from the constraint.
    generation: metadata.generation from the constraint.
  """
    constraintInfo = _messages.MessageField('ConstraintInfo', 1)
    creation = _messages.StringField(2)
    generation = _messages.IntegerField(3)