from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworksecurityOrganizationsLocationsAddressGroupsGetRequest(_messages.Message):
    """A NetworksecurityOrganizationsLocationsAddressGroupsGetRequest object.

  Fields:
    name: Required. A name of the AddressGroup to get. Must be in the format
      `projects/*/locations/{location}/addressGroups/*`.
  """
    name = _messages.StringField(1, required=True)