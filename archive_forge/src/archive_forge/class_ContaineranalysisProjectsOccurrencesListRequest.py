from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContaineranalysisProjectsOccurrencesListRequest(_messages.Message):
    """A ContaineranalysisProjectsOccurrencesListRequest object.

  Fields:
    filter: The filter expression.
    pageSize: Number of occurrences to return in the list. Must be positive.
      Max allowed page size is 1000. If not specified, page size defaults to
      20.
    pageToken: Token to provide to skip to a particular spot in the list.
    parent: Required. The name of the project to list occurrences for in the
      form of `projects/[PROJECT_ID]`.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)