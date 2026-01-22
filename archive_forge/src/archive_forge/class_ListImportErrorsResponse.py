from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListImportErrorsResponse(_messages.Message):
    """Response to `ListImportErrorsRequest`.

  Fields:
    importErrors: List of DAGs with statistics.
    nextPageToken: The page token used to query for the next page if one
      exists.
  """
    importErrors = _messages.MessageField('ImportError', 1, repeated=True)
    nextPageToken = _messages.StringField(2)