from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsRemoteworkersV1test2CommandTaskOutputs(_messages.Message):
    """Describes the expected outputs of the command.

  Fields:
    directories: A list of expected directories, relative to the execution
      root. All paths MUST be delimited by forward slashes.
    files: A list of expected files, relative to the execution root. All paths
      MUST be delimited by forward slashes.
    stderrDestination: The destination to which any stderr should be sent. The
      method by which the bot should send the stream contents to that
      destination is not defined in this API. As examples, the destination
      could be a file referenced in the `files` field in this message, or it
      could be a URI that must be written via the ByteStream API.
    stdoutDestination: The destination to which any stdout should be sent. The
      method by which the bot should send the stream contents to that
      destination is not defined in this API. As examples, the destination
      could be a file referenced in the `files` field in this message, or it
      could be a URI that must be written via the ByteStream API.
  """
    directories = _messages.StringField(1, repeated=True)
    files = _messages.StringField(2, repeated=True)
    stderrDestination = _messages.StringField(3)
    stdoutDestination = _messages.StringField(4)