from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AwsS3CompatibleData(_messages.Message):
    """An AwsS3CompatibleData resource.

  Fields:
    bucketName: Required. Specifies the name of the bucket.
    endpoint: Required. Specifies the endpoint of the storage service.
    path: Specifies the root path to transfer objects. Must be an empty string
      or full path name that ends with a '/'. This field is treated as an
      object prefix. As such, it should generally not begin with a '/'.
    region: Specifies the region to sign requests with. This can be left blank
      if requests should be signed with an empty region.
    s3Metadata: A S3 compatible metadata.
  """
    bucketName = _messages.StringField(1)
    endpoint = _messages.StringField(2)
    path = _messages.StringField(3)
    region = _messages.StringField(4)
    s3Metadata = _messages.MessageField('S3CompatibleMetadata', 5)