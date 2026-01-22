from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LoggingLocationsBucketsViewsDeleteRequest(_messages.Message):
    """A LoggingLocationsBucketsViewsDeleteRequest object.

  Fields:
    name: Required. The full resource name of the view to delete: "projects/[P
      ROJECT_ID]/locations/[LOCATION_ID]/buckets/[BUCKET_ID]/views/[VIEW_ID]"
      For example:"projects/my-project/locations/global/buckets/my-
      bucket/views/my-view"
  """
    name = _messages.StringField(1, required=True)