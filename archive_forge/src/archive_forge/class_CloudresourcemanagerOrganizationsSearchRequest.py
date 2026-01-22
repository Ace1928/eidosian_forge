from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudresourcemanagerOrganizationsSearchRequest(_messages.Message):
    """A CloudresourcemanagerOrganizationsSearchRequest object.

  Fields:
    pageSize: Optional. The maximum number of organizations to return in the
      response. The server can return fewer organizations than requested. If
      unspecified, server picks an appropriate default.
    pageToken: Optional. A pagination token returned from a previous call to
      `SearchOrganizations` that indicates from where listing should continue.
    query: Optional. An optional query string used to filter the Organizations
      to return in the response. Query rules are case-insensitive. ``` | Field
      | Description |
      |------------------|--------------------------------------------| |
      directoryCustomerId, owner.directoryCustomerId | Filters by directory
      customer id. | | domain | Filters by domain. | ``` Organizations may be
      queried by `directoryCustomerId` or by `domain`, where the domain is a G
      Suite domain, for example: * Query `directorycustomerid:123456789`
      returns Organization resources with `owner.directory_customer_id` equal
      to `123456789`. * Query `domain:google.com` returns Organization
      resources corresponding to the domain `google.com`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    query = _messages.StringField(3)