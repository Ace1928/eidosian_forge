from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EntityResult(_messages.Message):
    """The result of fetching an entity from Datastore.

  Fields:
    createTime: The time at which the entity was created. This field is set
      for `FULL` entity results. If this entity is missing, this field will
      not be set.
    cursor: A cursor that points to the position after the result entity. Set
      only when the `EntityResult` is part of a `QueryResultBatch` message.
    entity: The resulting entity.
    updateTime: The time at which the entity was last changed. This field is
      set for `FULL` entity results. If this entity is missing, this field
      will not be set.
    version: The version of the entity, a strictly positive number that
      monotonically increases with changes to the entity. This field is set
      for `FULL` entity results. For missing entities in `LookupResponse`,
      this is the version of the snapshot that was used to look up the entity,
      and it is always set except for eventually consistent reads.
  """
    createTime = _messages.StringField(1)
    cursor = _messages.BytesField(2)
    entity = _messages.MessageField('Entity', 3)
    updateTime = _messages.StringField(4)
    version = _messages.IntegerField(5)