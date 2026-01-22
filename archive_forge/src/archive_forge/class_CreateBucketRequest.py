from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CreateBucketRequest(_messages.Message):
    """The parameters to CreateBucket.

  Fields:
    bucket: Required. The new bucket. The region specified in the new bucket
      must be compliant with any Location Restriction Org Policy. The name
      field in the bucket is ignored.
    bucketId: Required. A client-assigned identifier such as "my-bucket".
      Identifiers are limited to 100 characters and can include only letters,
      digits, underscores, hyphens, and periods. Bucket identifiers must start
      with an alphanumeric character.
    parent: Required. The resource in which to create the log bucket:
      "projects/[PROJECT_ID]/locations/[LOCATION_ID]" For
      example:"projects/my-project/locations/global"
  """
    bucket = _messages.MessageField('LogBucket', 1)
    bucketId = _messages.StringField(2)
    parent = _messages.StringField(3)