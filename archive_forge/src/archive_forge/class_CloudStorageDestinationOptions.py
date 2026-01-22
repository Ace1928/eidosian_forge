from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudStorageDestinationOptions(_messages.Message):
    """Options to store reports in storage systems. Next ID: 3

  Fields:
    bucket: Destination bucket.
    destinationPath: Destination path is the path in the bucket where the
      report should be generated.
  """
    bucket = _messages.StringField(1)
    destinationPath = _messages.StringField(2)