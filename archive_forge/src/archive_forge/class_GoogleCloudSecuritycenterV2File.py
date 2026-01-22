from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV2File(_messages.Message):
    """File information about the related binary/library used by an executable,
  or the script used by a script interpreter

  Fields:
    contents: Prefix of the file contents as a JSON-encoded string.
    diskPath: Path of the file in terms of underlying disk/partition
      identifiers.
    hashedSize: The length in bytes of the file prefix that was hashed. If
      hashed_size == size, any hashes reported represent the entire file.
    partiallyHashed: True when the hash covers only a prefix of the file.
    path: Absolute path of the file as a JSON encoded string.
    sha256: SHA256 hash of the first hashed_size bytes of the file encoded as
      a hex string. If hashed_size == size, sha256 represents the SHA256 hash
      of the entire file.
    size: Size of the file in bytes.
  """
    contents = _messages.StringField(1)
    diskPath = _messages.MessageField('GoogleCloudSecuritycenterV2DiskPath', 2)
    hashedSize = _messages.IntegerField(3)
    partiallyHashed = _messages.BooleanField(4)
    path = _messages.StringField(5)
    sha256 = _messages.StringField(6)
    size = _messages.IntegerField(7)