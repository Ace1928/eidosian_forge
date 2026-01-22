from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1beta1GcsFilesetSpec(_messages.Message):
    """Describes a Cloud Storage fileset entry.

  Fields:
    filePatterns: Required. Patterns to identify a set of files in Google
      Cloud Storage. See [Cloud Storage documentation](https://cloud.google.co
      m/storage/docs/gsutil/addlhelp/WildcardNames) for more information. Note
      that bucket wildcards are currently not supported. Examples of valid
      file_patterns: * `gs://bucket_name/dir/*`: matches all files within
      `bucket_name/dir` directory. * `gs://bucket_name/dir/**`: matches all
      files in `bucket_name/dir` spanning all subdirectories. *
      `gs://bucket_name/file*`: matches files prefixed by `file` in
      `bucket_name` * `gs://bucket_name/??.txt`: matches files with two
      characters followed by `.txt` in `bucket_name` *
      `gs://bucket_name/[aeiou].txt`: matches files that contain a single
      vowel character followed by `.txt` in `bucket_name` *
      `gs://bucket_name/[a-m].txt`: matches files that contain `a`, `b`, ...
      or `m` followed by `.txt` in `bucket_name` * `gs://bucket_name/a/*/b`:
      matches all files in `bucket_name` that match `a/*/b` pattern, such as
      `a/c/b`, `a/d/b` * `gs://another_bucket/a.txt`: matches
      `gs://another_bucket/a.txt` You can combine wildcards to provide more
      powerful matches, for example: * `gs://bucket_name/[a-m]??.j*g`
    sampleGcsFileSpecs: Output only. Sample files contained in this fileset,
      not all files contained in this fileset are represented here.
  """
    filePatterns = _messages.StringField(1, repeated=True)
    sampleGcsFileSpecs = _messages.MessageField('GoogleCloudDatacatalogV1beta1GcsFileSpec', 2, repeated=True)