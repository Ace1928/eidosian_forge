from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuredlandingzoneOrganizationsLocationsOverwatchesListRequest(_messages.Message):
    """A SecuredlandingzoneOrganizationsLocationsOverwatchesListRequest object.

  Fields:
    pageSize: Optional. The maximum number of results to return in a single
      response. Default is 50, minimum is 1 and maximum is 1000.
    pageToken: Optional. The value returned by the last
      `ListOverwatchRequest`; indicates that this is a continuation of the
      prior `ListOverwatchRequest` call and that the system should return the
      next page of data.
    parent: Required. The name of the organization and region to list
      Overwatch resources. The format is
      organizations/{org_id}/locations/{location_id}.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)