from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsLocationsResultsListRequest(_messages.Message):
    """A CloudbuildProjectsLocationsResultsListRequest object.

  Fields:
    filter: Filter for the Records.
    pageSize: Size of the page to return. Default page_size = 50 Maximum
      page_size = 1000
    pageToken: Page start.
    parent: Required. The parent, which owns this collection of Results.
      Format: projects/{project}/locations/{location}/
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)