from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2Export(_messages.Message):
    """If set, the detailed data profiles will be persisted to the location of
  your choice whenever updated.

  Fields:
    profileTable: Store all table and column profiles in an existing table or
      a new table in an existing dataset. Each re-generation will result in a
      new row in BigQuery.
  """
    profileTable = _messages.MessageField('GooglePrivacyDlpV2BigQueryTable', 1)