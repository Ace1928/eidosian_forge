from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlOperationsListRequest(_messages.Message):
    """A SqlOperationsListRequest object.

  Fields:
    filter: Optional. A filter string that follows the rules of EBNF grammar
      (https://google.aip.dev/assets/misc/ebnf-filtering.txt). Cloud SQL
      provides filters for status, operationType, and startTime.
    instance: Cloud SQL instance ID. This does not include the project ID.
    maxResults: Maximum number of operations per response.
    pageToken: A previously-returned page token representing part of the
      larger set of results to view.
    project: Project ID of the project that contains the instance.
  """
    filter = _messages.StringField(1)
    instance = _messages.StringField(2)
    maxResults = _messages.IntegerField(3, variant=_messages.Variant.UINT32)
    pageToken = _messages.StringField(4)
    project = _messages.StringField(5, required=True)