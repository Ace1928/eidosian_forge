from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2DiscoveryBigQueryFilter(_messages.Message):
    """Determines what tables will have profiles generated within an
  organization or project. Includes the ability to filter by regular
  expression patterns on project ID, dataset ID, and table ID.

  Fields:
    otherTables: Catch-all. This should always be the last filter in the list
      because anything above it will apply first. Should only appear once in a
      configuration. If none is specified, a default one will be added
      automatically.
    tables: A specific set of tables for this filter to apply to. A table
      collection must be specified in only one filter per config. If a table
      id or dataset is empty, Cloud DLP assumes all tables in that collection
      must be profiled. Must specify a project ID.
  """
    otherTables = _messages.MessageField('GooglePrivacyDlpV2AllOtherBigQueryTables', 1)
    tables = _messages.MessageField('GooglePrivacyDlpV2BigQueryTableCollection', 2)