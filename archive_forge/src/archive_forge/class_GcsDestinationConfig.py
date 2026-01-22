from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GcsDestinationConfig(_messages.Message):
    """Google Cloud Storage destination configuration

  Fields:
    avroFileFormat: AVRO file format configuration.
    fileRotationInterval: The maximum duration for which new events are added
      before a file is closed and a new file is created. Values within the
      range of 15-60 seconds are allowed.
    fileRotationMb: The maximum file size to be saved in the bucket.
    jsonFileFormat: JSON file format configuration.
    path: Path inside the Cloud Storage bucket to write data to.
  """
    avroFileFormat = _messages.MessageField('AvroFileFormat', 1)
    fileRotationInterval = _messages.StringField(2)
    fileRotationMb = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    jsonFileFormat = _messages.MessageField('JsonFileFormat', 4)
    path = _messages.StringField(5)