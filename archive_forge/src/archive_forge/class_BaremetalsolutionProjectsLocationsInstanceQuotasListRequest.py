from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BaremetalsolutionProjectsLocationsInstanceQuotasListRequest(_messages.Message):
    """A BaremetalsolutionProjectsLocationsInstanceQuotasListRequest object.

  Fields:
    pageSize: Requested page size. The server might return fewer items than
      requested. If unspecified, server will pick an appropriate default.
      Notice that page_size field is not supported and won't be respected in
      the API request for now, will be updated when pagination is supported.
    pageToken: A token identifying a page of results from the server.
    parent: Required. Parent value for ListProvisioningQuotasRequest.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)