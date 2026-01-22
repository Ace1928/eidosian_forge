from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageManagedFoldersGetIamPolicyRequest(_messages.Message):
    """A StorageManagedFoldersGetIamPolicyRequest object.

  Fields:
    bucket: Name of the bucket containing the managed folder.
    managedFolder: The managed folder name/path.
    optionsRequestedPolicyVersion: The IAM policy format version to be
      returned. If the optionsRequestedPolicyVersion is for an older version
      that doesn't support part of the requested IAM policy, the request
      fails.
    userProject: The project to be billed for this request. Required for
      Requester Pays buckets.
  """
    bucket = _messages.StringField(1, required=True)
    managedFolder = _messages.StringField(2, required=True)
    optionsRequestedPolicyVersion = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    userProject = _messages.StringField(4)