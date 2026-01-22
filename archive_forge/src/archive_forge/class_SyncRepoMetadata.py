from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SyncRepoMetadata(_messages.Message):
    """Metadata of SyncRepo. This message is in the metadata field of
  Operation.

  Fields:
    name: The name of the repo being synchronized. Values are of the form
      `projects//repos/`.
    startTime: The time this operation is started.
    statusMessage: The latest status message on syncing the repository.
    updateTime: The time this operation's status message is updated.
  """
    name = _messages.StringField(1)
    startTime = _messages.StringField(2)
    statusMessage = _messages.StringField(3)
    updateTime = _messages.StringField(4)