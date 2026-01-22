from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatafusionProjectsLocationsVersionsListRequest(_messages.Message):
    """A DatafusionProjectsLocationsVersionsListRequest object.

  Fields:
    latestPatchOnly: Whether or not to return the latest patch of every
      available minor version. If true, only the latest patch will be
      returned. Ex. if allowed versions is [6.1.1, 6.1.2, 6.2.0] then response
      will be [6.1.2, 6.2.0]
    pageSize: The maximum number of items to return.
    pageToken: The next_page_token value to use if there are additional
      results to retrieve for this list request.
    parent: Required. The project and location for which to retrieve instance
      information in the format projects/{project}/locations/{location}.
  """
    latestPatchOnly = _messages.BooleanField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)