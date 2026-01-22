from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworksecurityOrganizationsLocationsAddressGroupsListReferencesRequest(_messages.Message):
    """A
  NetworksecurityOrganizationsLocationsAddressGroupsListReferencesRequest
  object.

  Fields:
    addressGroup: Required. A name of the AddressGroup to clone items to. Must
      be in the format
      `projects|organization/*/locations/{location}/addressGroups/*`.
    pageSize: The maximum number of references to return. If unspecified,
      server will pick an appropriate default. Server may return fewer items
      than requested. A caller should only rely on response's next_page_token
      to determine if there are more AddressGroupUsers left to be queried.
    pageToken: The next_page_token value returned from a previous List
      request, if any.
  """
    addressGroup = _messages.StringField(1, required=True)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)