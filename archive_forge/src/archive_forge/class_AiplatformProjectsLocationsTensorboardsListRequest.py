from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsTensorboardsListRequest(_messages.Message):
    """A AiplatformProjectsLocationsTensorboardsListRequest object.

  Fields:
    filter: Lists the Tensorboards that match the filter expression.
    orderBy: Field to use to sort the list.
    pageSize: The maximum number of Tensorboards to return. The service may
      return fewer than this value. If unspecified, at most 100 Tensorboards
      are returned. The maximum value is 100; values above 100 are coerced to
      100.
    pageToken: A page token, received from a previous
      TensorboardService.ListTensorboards call. Provide this to retrieve the
      subsequent page. When paginating, all other parameters provided to
      TensorboardService.ListTensorboards must match the call that provided
      the page token.
    parent: Required. The resource name of the Location to list Tensorboards.
      Format: `projects/{project}/locations/{location}`
    readMask: Mask specifying which fields to read.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)
    readMask = _messages.StringField(6)