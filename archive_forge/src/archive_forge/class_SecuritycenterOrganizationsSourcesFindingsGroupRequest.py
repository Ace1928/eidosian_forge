from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterOrganizationsSourcesFindingsGroupRequest(_messages.Message):
    """A SecuritycenterOrganizationsSourcesFindingsGroupRequest object.

  Fields:
    groupFindingsRequest: A GroupFindingsRequest resource to be passed as the
      request body.
    parent: Required. Name of the source to groupBy. If no location is
      specified, finding is assumed to be in global. The following list shows
      some examples: + `organizations/[organization_id]/sources/[source_id]` +
      `organizations/[organization_id]/sources/[source_id]/locations/[location
      _id]` + `folders/[folder_id]/sources/[source_id]` +
      `folders/[folder_id]/sources/[source_id]/locations/[location_id]` +
      `projects/[project_id]/sources/[source_id]` +
      `projects/[project_id]/sources/[source_id]/locations/[location_id]` To
      groupBy across all sources provide a source_id of `-`. The following
      list shows some examples: + `organizations/{organization_id}/sources/-`
      + `organizations/{organization_id}/sources/-/locations/[location_id]` +
      `folders/{folder_id}/sources/-` +
      `folders/{folder_id}/sources/-/locations/[location_id]` +
      `projects/{project_id}/sources/-` +
      `projects/{project_id}/sources/-/locations/[location_id]`
  """
    groupFindingsRequest = _messages.MessageField('GroupFindingsRequest', 1)
    parent = _messages.StringField(2, required=True)