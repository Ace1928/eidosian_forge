from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LinkedEntity(_messages.Message):
    """EntityMentions can be linked to multiple entities using a LinkedEntity
  message lets us add other fields, e.g. confidence.

  Fields:
    entityId: entity_id is a concept unique identifier. These are prefixed by
      a string that identifies the entity coding system, followed by the
      unique identifier within that system. For example, "UMLS/C0000970". This
      also supports ad hoc entities, which are formed by normalizing entity
      mention content.
  """
    entityId = _messages.StringField(1)