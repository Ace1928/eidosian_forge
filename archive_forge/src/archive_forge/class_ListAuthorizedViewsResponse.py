from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListAuthorizedViewsResponse(_messages.Message):
    """Response message for
  google.bigtable.admin.v2.BigtableTableAdmin.ListAuthorizedViews

  Fields:
    authorizedViews: The AuthorizedViews present in the requested table.
    nextPageToken: Set if not all tables could be returned in a single
      response. Pass this value to `page_token` in another request to get the
      next page of results.
  """
    authorizedViews = _messages.MessageField('AuthorizedView', 1, repeated=True)
    nextPageToken = _messages.StringField(2)