from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FetchLinkableRepositoriesResponse(_messages.Message):
    """Response message for FetchLinkableRepositories.

  Fields:
    nextPageToken: A token identifying a page of results the server should
      return.
    repositories: repositories ready to be created.
  """
    nextPageToken = _messages.StringField(1)
    repositories = _messages.MessageField('Repository', 2, repeated=True)