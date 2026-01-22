from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerProjectsInstancesDatabasesSessionsPartitionQueryRequest(_messages.Message):
    """A SpannerProjectsInstancesDatabasesSessionsPartitionQueryRequest object.

  Fields:
    partitionQueryRequest: A PartitionQueryRequest resource to be passed as
      the request body.
    session: Required. The session used to create the partitions.
  """
    partitionQueryRequest = _messages.MessageField('PartitionQueryRequest', 1)
    session = _messages.StringField(2, required=True)