from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2BigQueryTableCollection(_messages.Message):
    """Specifies a collection of BigQuery tables. Used for Discovery.

  Fields:
    includeRegexes: A collection of regular expressions to match a BigQuery
      table against.
  """
    includeRegexes = _messages.MessageField('GooglePrivacyDlpV2BigQueryRegexes', 1)