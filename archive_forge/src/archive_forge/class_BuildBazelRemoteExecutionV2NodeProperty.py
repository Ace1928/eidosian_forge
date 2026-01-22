from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BuildBazelRemoteExecutionV2NodeProperty(_messages.Message):
    """A single property for FileNodes, DirectoryNodes, and SymlinkNodes. The
  server is responsible for specifying the property `name`s that it accepts.
  If permitted by the server, the same `name` may occur multiple times.

  Fields:
    name: The property name.
    value: The property value.
  """
    name = _messages.StringField(1)
    value = _messages.StringField(2)