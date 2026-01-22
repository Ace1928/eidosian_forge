from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ListNatAddressesResponse(_messages.Message):
    """Response for ListNatAddresses.

  Fields:
    natAddresses: List of NAT Addresses for the instance.
    nextPageToken: Page token that you can include in a ListNatAddresses
      request to retrieve the next page of content. If omitted, no subsequent
      pages exist.
  """
    natAddresses = _messages.MessageField('GoogleCloudApigeeV1NatAddress', 1, repeated=True)
    nextPageToken = _messages.StringField(2)