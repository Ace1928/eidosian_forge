from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ExportTensorboardTimeSeriesDataRequest(_messages.Message):
    """Request message for TensorboardService.ExportTensorboardTimeSeriesData.

  Fields:
    filter: Exports the TensorboardTimeSeries' data that match the filter
      expression.
    orderBy: Field to use to sort the TensorboardTimeSeries' data. By default,
      TensorboardTimeSeries' data is returned in a pseudo random order.
    pageSize: The maximum number of data points to return per page. The
      default page_size is 1000. Values must be between 1 and 10000. Values
      above 10000 are coerced to 10000.
    pageToken: A page token, received from a previous
      ExportTensorboardTimeSeriesData call. Provide this to retrieve the
      subsequent page. When paginating, all other parameters provided to
      ExportTensorboardTimeSeriesData must match the call that provided the
      page token.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)