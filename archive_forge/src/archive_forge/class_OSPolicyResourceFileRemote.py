from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OSPolicyResourceFileRemote(_messages.Message):
    """Specifies a file available via some URI.

  Fields:
    sha256Checksum: SHA256 checksum of the remote file.
    uri: Required. URI from which to fetch the object. It should contain both
      the protocol and path following the format `{protocol}://{location}`.
  """
    sha256Checksum = _messages.StringField(1)
    uri = _messages.StringField(2)