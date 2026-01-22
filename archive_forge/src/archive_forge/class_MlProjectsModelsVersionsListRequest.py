from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MlProjectsModelsVersionsListRequest(_messages.Message):
    """A MlProjectsModelsVersionsListRequest object.

  Fields:
    filter: Optional. Specifies the subset of versions to retrieve.
    pageSize: Optional. The number of versions to retrieve per "page" of
      results. If there are more remaining results than this number, the
      response message will contain a valid value in the `next_page_token`
      field. The default value is 20, and the maximum page size is 100.
    pageToken: Optional. A page token to request the next page of results. You
      get the token from the `next_page_token` field of the response from the
      previous call.
    parent: Required. The name of the model for which to list the version.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)