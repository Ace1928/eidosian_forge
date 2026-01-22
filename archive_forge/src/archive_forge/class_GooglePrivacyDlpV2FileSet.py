from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2FileSet(_messages.Message):
    """Set of files to scan.

  Fields:
    regexFileSet: The regex-filtered set of files to scan. Exactly one of
      `url` or `regex_file_set` must be set.
    url: The Cloud Storage url of the file(s) to scan, in the format `gs:///`.
      Trailing wildcard in the path is allowed. If the url ends in a trailing
      slash, the bucket or directory represented by the url will be scanned
      non-recursively (content in sub-directories will not be scanned). This
      means that `gs://mybucket/` is equivalent to `gs://mybucket/*`, and
      `gs://mybucket/directory/` is equivalent to `gs://mybucket/directory/*`.
      Exactly one of `url` or `regex_file_set` must be set.
  """
    regexFileSet = _messages.MessageField('GooglePrivacyDlpV2CloudStorageRegexFileSet', 1)
    url = _messages.StringField(2)