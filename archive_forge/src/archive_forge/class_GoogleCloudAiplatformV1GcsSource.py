from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1GcsSource(_messages.Message):
    """The Google Cloud Storage location for the input content.

  Fields:
    uris: Required. Google Cloud Storage URI(-s) to the input file(s). May
      contain wildcards. For more information on wildcards, see
      https://cloud.google.com/storage/docs/gsutil/addlhelp/WildcardNames.
  """
    uris = _messages.StringField(1, repeated=True)