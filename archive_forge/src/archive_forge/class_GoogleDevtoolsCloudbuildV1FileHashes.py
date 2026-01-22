from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsCloudbuildV1FileHashes(_messages.Message):
    """Container message for hashes of byte content of files, used in
  SourceProvenance messages to verify integrity of source input to the build.

  Fields:
    fileHash: Collection of file hashes.
  """
    fileHash = _messages.MessageField('GoogleDevtoolsCloudbuildV1Hash', 1, repeated=True)