from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PathElement(_messages.Message):
    """A (kind, ID/name) pair used to construct a key path. If either name or
  ID is set, the element is complete. If neither is set, the element is
  incomplete.

  Fields:
    id: The auto-allocated ID of the entity. Never equal to zero. Values less
      than zero are discouraged and may not be supported in the future.
    kind: The kind of the entity. A kind matching regex `__.*__` is
      reserved/read-only. A kind must not contain more than 1500 bytes when
      UTF-8 encoded. Cannot be `""`. Must be valid UTF-8 bytes. Legacy values
      that are not valid UTF-8 are encoded as `__bytes__` where `` is the
      base-64 encoding of the bytes.
    name: The name of the entity. A name matching regex `__.*__` is
      reserved/read-only. A name must not be more than 1500 bytes when UTF-8
      encoded. Cannot be `""`. Must be valid UTF-8 bytes. Legacy values that
      are not valid UTF-8 are encoded as `__bytes__` where `` is the base-64
      encoding of the bytes.
  """
    id = _messages.IntegerField(1)
    kind = _messages.StringField(2)
    name = _messages.StringField(3)