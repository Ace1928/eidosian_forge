from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReadRequest(_messages.Message):
    """The request for Read and StreamingRead.

  Fields:
    columns: Required. The columns of table to be returned for each row
      matching this request.
    dataBoostEnabled: If this is for a partitioned read and this field is set
      to `true`, the request is executed with Spanner Data Boost independent
      compute resources. If the field is set to `true` but the request does
      not set `partition_token`, the API returns an `INVALID_ARGUMENT` error.
    directedReadOptions: Directed read options for this request.
    index: If non-empty, the name of an index on table. This index is used
      instead of the table primary key when interpreting key_set and sorting
      result rows. See key_set for further information.
    keySet: Required. `key_set` identifies the rows to be yielded. `key_set`
      names the primary keys of the rows in table to be yielded, unless index
      is present. If index is present, then key_set instead names index keys
      in index. If the partition_token field is empty, rows are yielded in
      table primary key order (if index is empty) or index key order (if index
      is non-empty). If the partition_token field is not empty, rows will be
      yielded in an unspecified order. It is not an error for the `key_set` to
      name rows that do not exist in the database. Read yields nothing for
      nonexistent rows.
    limit: If greater than zero, only the first `limit` rows are yielded. If
      `limit` is zero, the default is no limit. A limit cannot be specified if
      `partition_token` is set.
    partitionToken: If present, results will be restricted to the specified
      partition previously created using PartitionRead(). There must be an
      exact match for the values of fields common to this message and the
      PartitionReadRequest message used to create this partition_token.
    requestOptions: Common options for this request.
    resumeToken: If this request is resuming a previously interrupted read,
      `resume_token` should be copied from the last PartialResultSet yielded
      before the interruption. Doing this enables the new read to resume where
      the last read left off. The rest of the request parameters must exactly
      match the request that yielded this token.
    table: Required. The name of the table in the database to be read.
    transaction: The transaction to use. If none is provided, the default is a
      temporary read-only transaction with strong concurrency.
  """
    columns = _messages.StringField(1, repeated=True)
    dataBoostEnabled = _messages.BooleanField(2)
    directedReadOptions = _messages.MessageField('DirectedReadOptions', 3)
    index = _messages.StringField(4)
    keySet = _messages.MessageField('KeySet', 5)
    limit = _messages.IntegerField(6)
    partitionToken = _messages.BytesField(7)
    requestOptions = _messages.MessageField('RequestOptions', 8)
    resumeToken = _messages.BytesField(9)
    table = _messages.StringField(10)
    transaction = _messages.MessageField('TransactionSelector', 11)