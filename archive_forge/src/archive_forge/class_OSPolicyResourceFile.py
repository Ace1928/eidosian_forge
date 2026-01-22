from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OSPolicyResourceFile(_messages.Message):
    """A remote or local file.

  Fields:
    allowInsecure: Defaults to false. When false, files are subject to
      validations based on the file type: Remote: A checksum must be
      specified. Cloud Storage: An object generation number must be specified.
    gcs: A Cloud Storage object.
    localPath: A local path within the VM to use.
    remote: A generic remote file.
  """
    allowInsecure = _messages.BooleanField(1)
    gcs = _messages.MessageField('OSPolicyResourceFileGcs', 2)
    localPath = _messages.StringField(3)
    remote = _messages.MessageField('OSPolicyResourceFileRemote', 4)