from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2RecordTransformation(_messages.Message):
    """The field in a record to transform.

  Fields:
    containerTimestamp: Findings container modification timestamp, if
      applicable.
    containerVersion: Container version, if available ("generation" for Cloud
      Storage).
    fieldId: For record transformations, provide a field.
  """
    containerTimestamp = _messages.StringField(1)
    containerVersion = _messages.StringField(2)
    fieldId = _messages.MessageField('GooglePrivacyDlpV2FieldId', 3)