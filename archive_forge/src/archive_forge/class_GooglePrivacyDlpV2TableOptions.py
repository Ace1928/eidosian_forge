from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2TableOptions(_messages.Message):
    """Instructions regarding the table content being inspected.

  Fields:
    identifyingFields: The columns that are the primary keys for table objects
      included in ContentItem. A copy of this cell's value will stored
      alongside alongside each finding so that the finding can be traced to
      the specific row it came from. No more than 3 may be provided.
  """
    identifyingFields = _messages.MessageField('GooglePrivacyDlpV2FieldId', 1, repeated=True)