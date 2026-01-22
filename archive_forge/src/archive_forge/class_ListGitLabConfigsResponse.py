from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListGitLabConfigsResponse(_messages.Message):
    """RPC response object returned by ListGitLabConfigs RPC method.

  Fields:
    gitlabConfigs: A list of GitLabConfigs
    nextPageToken: A token that can be sent as `page_token` to retrieve the
      next page If this field is omitted, there are no subsequent pages.
  """
    gitlabConfigs = _messages.MessageField('GitLabConfig', 1, repeated=True)
    nextPageToken = _messages.StringField(2)