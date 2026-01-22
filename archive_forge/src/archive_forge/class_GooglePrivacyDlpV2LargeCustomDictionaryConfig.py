from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2LargeCustomDictionaryConfig(_messages.Message):
    """Configuration for a custom dictionary created from a data source of any
  size up to the maximum size defined in the
  [limits](https://cloud.google.com/sensitive-data-protection/limits) page.
  The artifacts of dictionary creation are stored in the specified Cloud
  Storage location. Consider using `CustomInfoType.Dictionary` for smaller
  dictionaries that satisfy the size requirements.

  Fields:
    bigQueryField: Field in a BigQuery table where each cell represents a
      dictionary phrase.
    cloudStorageFileSet: Set of files containing newline-delimited lists of
      dictionary phrases.
    outputPath: Location to store dictionary artifacts in Cloud Storage. These
      files will only be accessible by project owners and the DLP API. If any
      of these artifacts are modified, the dictionary is considered invalid
      and can no longer be used.
  """
    bigQueryField = _messages.MessageField('GooglePrivacyDlpV2BigQueryField', 1)
    cloudStorageFileSet = _messages.MessageField('GooglePrivacyDlpV2CloudStorageFileSet', 2)
    outputPath = _messages.MessageField('GooglePrivacyDlpV2CloudStoragePath', 3)