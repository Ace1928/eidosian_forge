from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasPortalNode(_messages.Message):
    """The Node.

  Fields:
    displayName: The node's display name.
    name: Output only. Resource name.
    sasUserIds: User ids used by the devices belonging to this node.
  """
    displayName = _messages.StringField(1)
    name = _messages.StringField(2)
    sasUserIds = _messages.StringField(3, repeated=True)