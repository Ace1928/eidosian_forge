from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import extra_types
class FusiontablesTemplateListRequest(_messages.Message):
    """A FusiontablesTemplateListRequest object.

  Fields:
    maxResults: Maximum number of templates to return. Optional. Default is 5.
    pageToken: Continuation token specifying which results page to return.
      Optional.
    tableId: Identifier for the table whose templates are being requested
  """
    maxResults = _messages.IntegerField(1, variant=_messages.Variant.UINT32)
    pageToken = _messages.StringField(2)
    tableId = _messages.StringField(3, required=True)