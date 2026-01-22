from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1beta2DocumentEntityRelation(_messages.Message):
    """Relationship between Entities.

  Fields:
    objectId: Object entity id.
    relation: Relationship description.
    subjectId: Subject entity id.
  """
    objectId = _messages.StringField(1)
    relation = _messages.StringField(2)
    subjectId = _messages.StringField(3)