from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworksecurityOrganizationsLocationsAddressGroupsRemoveItemsRequest(_messages.Message):
    """A NetworksecurityOrganizationsLocationsAddressGroupsRemoveItemsRequest
  object.

  Fields:
    addressGroup: Required. A name of the AddressGroup to remove items from.
      Must be in the format
      `projects|organization/*/locations/{location}/addressGroups/*`.
    removeAddressGroupItemsRequest: A RemoveAddressGroupItemsRequest resource
      to be passed as the request body.
  """
    addressGroup = _messages.StringField(1, required=True)
    removeAddressGroupItemsRequest = _messages.MessageField('RemoveAddressGroupItemsRequest', 2)