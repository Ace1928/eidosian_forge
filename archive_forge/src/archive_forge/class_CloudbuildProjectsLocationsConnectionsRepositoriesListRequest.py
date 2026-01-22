from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsLocationsConnectionsRepositoriesListRequest(_messages.Message):
    """A CloudbuildProjectsLocationsConnectionsRepositoriesListRequest object.

  Fields:
    filter: A filter expression that filters resources listed in the response.
      Expressions must follow API improvement proposal
      [AIP-160](https://google.aip.dev/160). e.g.
      `remote_uri:"https://github.com*"`.
    pageSize: Number of results to return in the list.
    pageToken: Page start.
    parent: Required. The parent, which owns this collection of Repositories.
      Format: `projects/*/locations/*/connections/*`.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)