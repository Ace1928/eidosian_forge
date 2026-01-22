from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsLocationsConnectionsListRequest(_messages.Message):
    """A CloudbuildProjectsLocationsConnectionsListRequest object.

  Fields:
    pageSize: Number of results to return in the list.
    pageToken: Page start.
    parent: Required. The parent, which owns this collection of Connections.
      Format: `projects/*/locations/*`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)