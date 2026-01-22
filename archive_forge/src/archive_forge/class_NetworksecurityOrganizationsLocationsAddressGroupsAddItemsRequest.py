from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworksecurityOrganizationsLocationsAddressGroupsAddItemsRequest(_messages.Message):
    """A NetworksecurityOrganizationsLocationsAddressGroupsAddItemsRequest
  object.

  Fields:
    addAddressGroupItemsRequest: A AddAddressGroupItemsRequest resource to be
      passed as the request body.
    addressGroup: Required. A name of the AddressGroup to add items to. Must
      be in the format
      `projects|organization/*/locations/{location}/addressGroups/*`.
  """
    addAddressGroupItemsRequest = _messages.MessageField('AddAddressGroupItemsRequest', 1)
    addressGroup = _messages.StringField(2, required=True)