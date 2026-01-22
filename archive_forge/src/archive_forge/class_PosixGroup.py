from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PosixGroup(_messages.Message):
    """POSIX Group definition to represent a group in a POSIX compliant system.

  Fields:
    gid: GID of the POSIX group.
    name: Name of the POSIX group.
    systemId: System identifier for which group name and gid apply to. If not
      specified it will default to empty value.
  """
    gid = _messages.IntegerField(1, variant=_messages.Variant.UINT64)
    name = _messages.StringField(2)
    systemId = _messages.StringField(3)