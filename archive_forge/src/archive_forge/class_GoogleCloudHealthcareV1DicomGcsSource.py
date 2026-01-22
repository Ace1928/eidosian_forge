from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudHealthcareV1DicomGcsSource(_messages.Message):
    """Specifies the configuration for importing data from Cloud Storage.

  Fields:
    uri: Points to a Cloud Storage URI containing file(s) with content only.
      The URI must be in the following format: `gs://{bucket_id}/{object_id}`.
      The URI can include wildcards in `object_id` and thus identify multiple
      files. Supported wildcards: * '*' to match 0 or more non-separator
      characters * '**' to match 0 or more characters (including separators).
      Must be used at the end of a path and with no other wildcards in the
      path. Can also be used with a file extension (such as .dcm), which
      imports all files with the extension in the specified directory and its
      sub-directories. For example, `gs://my-bucket/my-directory/**.dcm`
      imports all files with .dcm extensions in `my-directory/` and its sub-
      directories. * '?' to match 1 character. All other URI formats are
      invalid. Files matching the wildcard are expected to contain content
      only, no metadata.
  """
    uri = _messages.StringField(1)