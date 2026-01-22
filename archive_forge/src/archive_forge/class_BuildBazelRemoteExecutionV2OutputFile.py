from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BuildBazelRemoteExecutionV2OutputFile(_messages.Message):
    """An `OutputFile` is similar to a FileNode, but it is used as an output in
  an `ActionResult`. It allows a full file path rather than only a name.

  Fields:
    contents: The contents of the file if inlining was requested. The server
      SHOULD NOT inline file contents unless requested by the client in the
      GetActionResultRequest message. The server MAY omit inlining, even if
      requested, and MUST do so if inlining would cause the response to exceed
      message size limits. Clients SHOULD NOT populate this field when
      uploading to the cache.
    digest: The digest of the file's content.
    isExecutable: True if file is executable, false otherwise.
    nodeProperties: A BuildBazelRemoteExecutionV2NodeProperties attribute.
    path: The full path of the file relative to the working directory,
      including the filename. The path separator is a forward slash `/`. Since
      this is a relative path, it MUST NOT begin with a leading forward slash.
  """
    contents = _messages.BytesField(1)
    digest = _messages.MessageField('BuildBazelRemoteExecutionV2Digest', 2)
    isExecutable = _messages.BooleanField(3)
    nodeProperties = _messages.MessageField('BuildBazelRemoteExecutionV2NodeProperties', 4)
    path = _messages.StringField(5)