from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EntityMentionRelationship(_messages.Message):
    """Defines directed relationship from one entity mention to another.

  Fields:
    confidence: The model's confidence in this annotation. A number between 0
      and 1.
    objectId: object_id is the id of the object entity mention.
    subjectId: subject_id is the id of the subject entity mention.
  """
    confidence = _messages.FloatField(1)
    objectId = _messages.StringField(2)
    subjectId = _messages.StringField(3)