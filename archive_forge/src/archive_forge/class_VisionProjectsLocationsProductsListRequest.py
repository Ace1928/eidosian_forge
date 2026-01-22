from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VisionProjectsLocationsProductsListRequest(_messages.Message):
    """A VisionProjectsLocationsProductsListRequest object.

  Fields:
    pageSize: The maximum number of items to return. Default 10, maximum 100.
    pageToken: The next_page_token returned from a previous List request, if
      any.
    parent: Required. The project OR ProductSet from which Products should be
      listed. Format: `projects/PROJECT_ID/locations/LOC_ID`
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)