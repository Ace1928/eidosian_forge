from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SearchDeploymentRevisionsResponse(_messages.Message):
    """Response object for `SearchDeploymentRevisions`.

  Fields:
    deployments: The list of requested deployment revisions.
    nextPageToken: A token that can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
  """
    deployments = _messages.MessageField('Deployment', 1, repeated=True)
    nextPageToken = _messages.StringField(2)