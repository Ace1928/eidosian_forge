from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsDataAttributeBindingsListRequest(_messages.Message):
    """A DataplexProjectsLocationsDataAttributeBindingsListRequest object.

  Fields:
    filter: Optional. Filter request. Filter using resource:
      filter=resource:"resource-name" Filter using attribute:
      filter=attributes:"attribute-name" Filter using attribute in paths list:
      filter=paths.attributes:"attribute-name"
    orderBy: Optional. Order by fields for the result.
    pageSize: Optional. Maximum number of DataAttributeBindings to return. The
      service may return fewer than this value. If unspecified, at most 10
      DataAttributeBindings will be returned. The maximum value is 1000;
      values above 1000 will be coerced to 1000.
    pageToken: Optional. Page token received from a previous
      ListDataAttributeBindings call. Provide this to retrieve the subsequent
      page. When paginating, all other parameters provided to
      ListDataAttributeBindings must match the call that provided the page
      token.
    parent: Required. The resource name of the Location:
      projects/{project_number}/locations/{location_id}
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)