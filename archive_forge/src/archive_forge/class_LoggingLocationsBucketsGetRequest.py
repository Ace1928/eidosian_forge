from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LoggingLocationsBucketsGetRequest(_messages.Message):
    """A LoggingLocationsBucketsGetRequest object.

  Fields:
    name: Required. The resource name of the bucket:
      "projects/[PROJECT_ID]/locations/[LOCATION_ID]/buckets/[BUCKET_ID]" "org
      anizations/[ORGANIZATION_ID]/locations/[LOCATION_ID]/buckets/[BUCKET_ID]
      " "billingAccounts/[BILLING_ACCOUNT_ID]/locations/[LOCATION_ID]/buckets/
      [BUCKET_ID]"
      "folders/[FOLDER_ID]/locations/[LOCATION_ID]/buckets/[BUCKET_ID]" For
      example:"projects/my-project/locations/global/buckets/my-bucket"
  """
    name = _messages.StringField(1, required=True)