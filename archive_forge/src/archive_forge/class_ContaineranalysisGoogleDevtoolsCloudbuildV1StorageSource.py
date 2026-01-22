from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContaineranalysisGoogleDevtoolsCloudbuildV1StorageSource(_messages.Message):
    """Location of the source in an archive file in Cloud Storage.

  Enums:
    SourceFetcherValueValuesEnum: Optional. Option to specify the tool to
      fetch the source file for the build.

  Fields:
    bucket: Cloud Storage bucket containing the source (see [Bucket Name
      Requirements](https://cloud.google.com/storage/docs/bucket-
      naming#requirements)).
    generation: Cloud Storage generation for the object. If the generation is
      omitted, the latest generation will be used.
    object: Cloud Storage object containing the source. This object must be a
      zipped (`.zip`) or gzipped archive file (`.tar.gz`) containing source to
      build.
    sourceFetcher: Optional. Option to specify the tool to fetch the source
      file for the build.
  """

    class SourceFetcherValueValuesEnum(_messages.Enum):
        """Optional. Option to specify the tool to fetch the source file for the
    build.

    Values:
      SOURCE_FETCHER_UNSPECIFIED: Unspecified defaults to GSUTIL.
      GSUTIL: Use the "gsutil" tool to download the source file.
      GCS_FETCHER: Use the Cloud Storage Fetcher tool to download the source
        file.
    """
        SOURCE_FETCHER_UNSPECIFIED = 0
        GSUTIL = 1
        GCS_FETCHER = 2
    bucket = _messages.StringField(1)
    generation = _messages.IntegerField(2)
    object = _messages.StringField(3)
    sourceFetcher = _messages.EnumField('SourceFetcherValueValuesEnum', 4)