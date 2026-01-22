from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SoftwareRecipeArtifactGcs(_messages.Message):
    """Specifies an artifact available as a Google Cloud Storage object.

  Fields:
    bucket: Bucket of the Google Cloud Storage object. Given an example URL:
      `https://storage.googleapis.com/my-bucket/foo/bar#1234567` this value
      would be `my-bucket`.
    generation: Must be provided if allow_insecure is false. Generation number
      of the Google Cloud Storage object. `https://storage.googleapis.com/my-
      bucket/foo/bar#1234567` this value would be `1234567`.
    object: Name of the Google Cloud Storage object. As specified [here]
      (https://cloud.google.com/storage/docs/naming#objectnames) Given an
      example URL: `https://storage.googleapis.com/my-bucket/foo/bar#1234567`
      this value would be `foo/bar`.
  """
    bucket = _messages.StringField(1)
    generation = _messages.IntegerField(2)
    object = _messages.StringField(3)