from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerProjectsInstancesDatabasesSessionsPartitionReadRequest(_messages.Message):
    """A SpannerProjectsInstancesDatabasesSessionsPartitionReadRequest object.

  Fields:
    partitionReadRequest: A PartitionReadRequest resource to be passed as the
      request body.
    session: Required. The session used to create the partitions.
  """
    partitionReadRequest = _messages.MessageField('PartitionReadRequest', 1)
    session = _messages.StringField(2, required=True)