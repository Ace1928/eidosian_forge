from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2Table(_messages.Message):
    """Structured content to inspect. Up to 50,000 `Value`s per request
  allowed. See https://cloud.google.com/sensitive-data-
  protection/docs/inspecting-structured-text#inspecting_a_table to learn more.

  Fields:
    headers: Headers of the table.
    rows: Rows of the table.
  """
    headers = _messages.MessageField('GooglePrivacyDlpV2FieldId', 1, repeated=True)
    rows = _messages.MessageField('GooglePrivacyDlpV2Row', 2, repeated=True)