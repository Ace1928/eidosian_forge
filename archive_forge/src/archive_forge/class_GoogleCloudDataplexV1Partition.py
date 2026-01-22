from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1Partition(_messages.Message):
    """Represents partition metadata contained within entity instances.

  Fields:
    etag: Optional. The etag for this partition.
    location: Required. Immutable. The location of the entity data within the
      partition, for example,
      gs://bucket/path/to/entity/key1=value1/key2=value2. Or
      projects//datasets//tables/
    name: Output only. Partition values used in the HTTP URL must be double
      encoded. For example, url_encode(url_encode(value)) can be used to
      encode "US:CA/CA#Sunnyvale so that the request URL ends with
      "/partitions/US%253ACA/CA%2523Sunnyvale". The name field in the response
      retains the encoded format.
    values: Required. Immutable. The set of values representing the partition,
      which correspond to the partition schema defined in the parent entity.
  """
    etag = _messages.StringField(1)
    location = _messages.StringField(2)
    name = _messages.StringField(3)
    values = _messages.StringField(4, repeated=True)