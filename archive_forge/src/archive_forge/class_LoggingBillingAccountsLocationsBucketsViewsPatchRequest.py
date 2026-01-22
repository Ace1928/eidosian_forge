from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LoggingBillingAccountsLocationsBucketsViewsPatchRequest(_messages.Message):
    """A LoggingBillingAccountsLocationsBucketsViewsPatchRequest object.

  Fields:
    logView: A LogView resource to be passed as the request body.
    name: Required. The full resource name of the view to update "projects/[PR
      OJECT_ID]/locations/[LOCATION_ID]/buckets/[BUCKET_ID]/views/[VIEW_ID]"
      For example:"projects/my-project/locations/global/buckets/my-
      bucket/views/my-view"
    updateMask: Optional. Field mask that specifies the fields in view that
      need an update. A field will be overwritten if, and only if, it is in
      the update mask. name and output only fields cannot be updated.For a
      detailed FieldMask definition, see
      https://developers.google.com/protocol-
      buffers/docs/reference/google.protobuf#google.protobuf.FieldMaskFor
      example: updateMask=filter
  """
    logView = _messages.MessageField('LogView', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)