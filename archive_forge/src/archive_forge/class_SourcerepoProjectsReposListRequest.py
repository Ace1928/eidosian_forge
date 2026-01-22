from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SourcerepoProjectsReposListRequest(_messages.Message):
    """A SourcerepoProjectsReposListRequest object.

  Fields:
    name: The project ID whose repos should be listed. Values are of the form
      `projects/`.
    pageSize: Maximum number of repositories to return; between 1 and 500. If
      not set or zero, defaults to 100 at the server.
    pageToken: Resume listing repositories where a prior ListReposResponse
      left off. This is an opaque token that must be obtained from a recent,
      prior ListReposResponse's next_page_token field.
  """
    name = _messages.StringField(1, required=True)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)