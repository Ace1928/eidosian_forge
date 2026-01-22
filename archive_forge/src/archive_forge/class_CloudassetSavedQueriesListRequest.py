from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudassetSavedQueriesListRequest(_messages.Message):
    """A CloudassetSavedQueriesListRequest object.

  Fields:
    filter: Optional. The expression to filter resources. The expression is a
      list of zero or more restrictions combined via logical operators `AND`
      and `OR`. When `AND` and `OR` are both used in the expression,
      parentheses must be appropriately used to group the combinations. The
      expression may also contain regular expressions. See
      https://google.aip.dev/160 for more information on the grammar.
    pageSize: Optional. The maximum number of saved queries to return per
      page. The service may return fewer than this value. If unspecified, at
      most 50 will be returned. The maximum value is 1000; values above 1000
      will be coerced to 1000.
    pageToken: Optional. A page token, received from a previous
      `ListSavedQueries` call. Provide this to retrieve the subsequent page.
      When paginating, all other parameters provided to `ListSavedQueries`
      must match the call that provided the page token.
    parent: Required. The parent project/folder/organization whose
      savedQueries are to be listed. It can only be using
      project/folder/organization number (such as "folders/12345")", or a
      project ID (such as "projects/my-project-id").
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)