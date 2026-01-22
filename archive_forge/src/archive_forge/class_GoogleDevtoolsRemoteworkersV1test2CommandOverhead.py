from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsRemoteworkersV1test2CommandOverhead(_messages.Message):
    """DEPRECATED - use CommandResult instead. Can be used as part of
  CompleteRequest.metadata, or are part of a more sophisticated message.

  Fields:
    duration: The elapsed time between calling Accept and Complete. The server
      will also have its own idea of what this should be, but this excludes
      the overhead of the RPCs and the bot response time.
    overhead: The amount of time *not* spent executing the command (ie
      uploading/downloading files).
  """
    duration = _messages.StringField(1)
    overhead = _messages.StringField(2)