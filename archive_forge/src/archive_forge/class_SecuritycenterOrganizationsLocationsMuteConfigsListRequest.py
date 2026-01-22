from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterOrganizationsLocationsMuteConfigsListRequest(_messages.Message):
    """A SecuritycenterOrganizationsLocationsMuteConfigsListRequest object.

  Fields:
    pageSize: The maximum number of configs to return. The service may return
      fewer than this value. If unspecified, at most 10 configs will be
      returned. The maximum value is 1000; values above 1000 will be coerced
      to 1000.
    pageToken: A page token, received from a previous `ListMuteConfigs` call.
      Provide this to retrieve the subsequent page. When paginating, all other
      parameters provided to `ListMuteConfigs` must match the call that
      provided the page token.
    parent: Required. The parent, which owns the collection of mute configs.
      Its format is "organizations/[organization_id]", "folders/[folder_id]",
      "projects/[project_id]",
      "organizations/[organization_id]/locations/[location_id]",
      "folders/[folder_id]/locations/[location_id]",
      "projects/[project_id]/locations/[location_id]".
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)