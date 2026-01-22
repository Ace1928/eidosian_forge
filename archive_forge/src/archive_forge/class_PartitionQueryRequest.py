from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PartitionQueryRequest(_messages.Message):
    """The request for Firestore.PartitionQuery.

  Fields:
    pageSize: The maximum number of partitions to return in this call, subject
      to `partition_count`. For example, if `partition_count` = 10 and
      `page_size` = 8, the first call to PartitionQuery will return up to 8
      partitions and a `next_page_token` if more results exist. A second call
      to PartitionQuery will return up to 2 partitions, to complete the total
      of 10 specified in `partition_count`.
    pageToken: The `next_page_token` value returned from a previous call to
      PartitionQuery that may be used to get an additional set of results.
      There are no ordering guarantees between sets of results. Thus, using
      multiple sets of results will require merging the different result sets.
      For example, two subsequent calls using a page_token may return: *
      cursor B, cursor M, cursor Q * cursor A, cursor U, cursor W To obtain a
      complete result set ordered with respect to the results of the query
      supplied to PartitionQuery, the results sets should be merged: cursor A,
      cursor B, cursor M, cursor Q, cursor U, cursor W
    partitionCount: The desired maximum number of partition points. The
      partitions may be returned across multiple pages of results. The number
      must be positive. The actual number of partitions returned may be fewer.
      For example, this may be set to one fewer than the number of parallel
      queries to be run, or in running a data pipeline job, one fewer than the
      number of workers or compute instances available.
    readTime: Reads documents as they were at the given time. This must be a
      microsecond precision timestamp within the past one hour, or if Point-
      in-Time Recovery is enabled, can additionally be a whole minute
      timestamp within the past 7 days.
    structuredQuery: A structured query. Query must specify collection with
      all descendants and be ordered by name ascending. Other filters, order
      bys, limits, offsets, and start/end cursors are not supported.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    partitionCount = _messages.IntegerField(3)
    readTime = _messages.StringField(4)
    structuredQuery = _messages.MessageField('StructuredQuery', 5)