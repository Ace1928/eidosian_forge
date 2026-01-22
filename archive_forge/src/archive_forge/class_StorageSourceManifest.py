from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageSourceManifest(_messages.Message):
    """Location of the source manifest in Google Cloud Storage. This feature is
  in Preview; see description
  [here](https://github.com/GoogleCloudPlatform/cloud-
  builders/tree/master/gcs-fetcher).

  Fields:
    bucket: Google Cloud Storage bucket containing the source manifest (see
      [Bucket Name Requirements](https://cloud.google.com/storage/docs/bucket-
      naming#requirements)).
    generation: Google Cloud Storage generation for the object. If the
      generation is omitted, the latest generation will be used.
    object: Google Cloud Storage object containing the source manifest. This
      object must be a JSON file.
  """
    bucket = _messages.StringField(1)
    generation = _messages.IntegerField(2)
    object = _messages.StringField(3)