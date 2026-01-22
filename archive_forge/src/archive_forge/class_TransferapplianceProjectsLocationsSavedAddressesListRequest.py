from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TransferapplianceProjectsLocationsSavedAddressesListRequest(_messages.Message):
    """A TransferapplianceProjectsLocationsSavedAddressesListRequest object.

  Fields:
    filter: Filtering results. See https://google.aip.dev/160 for more
      details.
    orderBy: Field to sort by. See https://google.aip.dev/132#ordering for
      more details.
    pageSize: Requested page size. Server may return fewer items than
      requested. If unspecified, server will pick the default size of 500.
      Maximum allowed page_size is 1000.
    pageToken: A token identifying a page of results the server should return.
    parent: Required. Parent value for ListSavedAddressesRequest.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)