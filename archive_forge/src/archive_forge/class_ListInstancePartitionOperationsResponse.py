from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListInstancePartitionOperationsResponse(_messages.Message):
    """The response for ListInstancePartitionOperations.

  Fields:
    nextPageToken: `next_page_token` can be sent in a subsequent
      ListInstancePartitionOperations call to fetch more of the matching
      metadata.
    operations: The list of matching instance partition long-running
      operations. Each operation's name will be prefixed by the instance
      partition's name. The operation's metadata field type
      `metadata.type_url` describes the type of the metadata.
    unreachableInstancePartitions: The list of unreachable instance
      partitions. It includes the names of instance partitions whose operation
      metadata could not be retrieved within instance_partition_deadline.
  """
    nextPageToken = _messages.StringField(1)
    operations = _messages.MessageField('Operation', 2, repeated=True)
    unreachableInstancePartitions = _messages.StringField(3, repeated=True)