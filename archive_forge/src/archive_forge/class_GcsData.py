from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GcsData(_messages.Message):
    """In a GcsData resource, an object's name is the Cloud Storage object's
  name and its "last modification time" refers to the object's `updated`
  property of Cloud Storage objects, which changes when the content or the
  metadata of the object is updated.

  Fields:
    bucketName: Required. Cloud Storage bucket name. Must meet [Bucket Name
      Requirements](/storage/docs/naming#requirements).
    managedFolderTransferEnabled: Preview. Enables the transfer of managed
      folders between Cloud Storage buckets. Set this option on the
      gcs_data_source. If set to true: - Managed folders in the source bucket
      are transferred to the destination bucket. - Managed folders in the
      destination bucket are overwritten. Other OVERWRITE options are not
      supported. See [Transfer Cloud Storage managed folders](/storage-
      transfer/docs/managed-folders).
    path: Root path to transfer objects. Must be an empty string or full path
      name that ends with a '/'. This field is treated as an object prefix. As
      such, it should generally not begin with a '/'. The root path value must
      meet [Object Name Requirements](/storage/docs/naming#objectnames).
  """
    bucketName = _messages.StringField(1)
    managedFolderTransferEnabled = _messages.BooleanField(2)
    path = _messages.StringField(3)