from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2BigQueryRegex(_messages.Message):
    """A pattern to match against one or more tables, datasets, or projects
  that contain BigQuery tables. At least one pattern must be specified.
  Regular expressions use RE2
  [syntax](https://github.com/google/re2/wiki/Syntax); a guide can be found
  under the google/re2 repository on GitHub.

  Fields:
    datasetIdRegex: If unset, this property matches all datasets.
    projectIdRegex: For organizations, if unset, will match all projects. Has
      no effect for data profile configurations created within a project.
    tableIdRegex: If unset, this property matches all tables.
  """
    datasetIdRegex = _messages.StringField(1)
    projectIdRegex = _messages.StringField(2)
    tableIdRegex = _messages.StringField(3)