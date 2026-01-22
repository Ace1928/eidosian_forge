from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsRemoteworkersV1test2FileMetadata(_messages.Message):
    """The metadata for a file. Similar to the equivalent message in the Remote
  Execution API.

  Fields:
    contents: If the file is small enough, its contents may also or
      alternatively be listed here.
    digest: A pointer to the contents of the file. The method by which a
      client retrieves the contents from a CAS system is not defined here.
    isExecutable: Properties of the file
    path: The path of this file. If this message is part of the
      CommandOutputs.outputs fields, the path is relative to the execution
      root and must correspond to an entry in CommandTask.outputs.files. If
      this message is part of a Directory message, then the path is relative
      to the root of that directory. All paths MUST be delimited by forward
      slashes.
  """
    contents = _messages.BytesField(1)
    digest = _messages.MessageField('GoogleDevtoolsRemoteworkersV1test2Digest', 2)
    isExecutable = _messages.BooleanField(3)
    path = _messages.StringField(4)