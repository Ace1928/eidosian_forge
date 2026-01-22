from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GcsOutputDestination(_messages.Message):
    """The Google Cloud Storage location for the output content.

  Fields:
    outputUriPrefix: Required. Google Cloud Storage URI to output directory.
      For example, `gs://bucket/directory`. The requesting user must have
      write permission to the bucket. The directory will be created if it
      doesn't exist.
  """
    outputUriPrefix = _messages.StringField(1)