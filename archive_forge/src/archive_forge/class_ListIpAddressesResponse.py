from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListIpAddressesResponse(_messages.Message):
    """The response of listing `IpAddress` objects in a given `ClusterGroup`.

  Fields:
    ipAddresses: A list of `IpAddress` objects.
    nextPageToken: A token, which can be send as `page_token` to retrieve the
      next page. If you omit this field, there are no subsequent pages.
  """
    ipAddresses = _messages.MessageField('IpAddress', 1, repeated=True)
    nextPageToken = _messages.StringField(2)