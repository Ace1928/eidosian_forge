from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BuildBazelRemoteExecutionV2FileNode(_messages.Message):
    """A `FileNode` represents a single file and associated metadata.

  Fields:
    digest: The digest of the file's content.
    isExecutable: True if file is executable, false otherwise.
    name: The name of the file.
    nodeProperties: A BuildBazelRemoteExecutionV2NodeProperties attribute.
  """
    digest = _messages.MessageField('BuildBazelRemoteExecutionV2Digest', 1)
    isExecutable = _messages.BooleanField(2)
    name = _messages.StringField(3)
    nodeProperties = _messages.MessageField('BuildBazelRemoteExecutionV2NodeProperties', 4)