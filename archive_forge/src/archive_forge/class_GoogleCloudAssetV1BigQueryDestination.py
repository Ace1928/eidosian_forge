from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAssetV1BigQueryDestination(_messages.Message):
    """A BigQuery destination.

  Enums:
    PartitionKeyValueValuesEnum: The partition key for BigQuery partitioned
      table.

  Fields:
    dataset: Required. The BigQuery dataset in format
      "projects/projectId/datasets/datasetId", to which the analysis results
      should be exported. If this dataset does not exist, the export call will
      return an INVALID_ARGUMENT error.
    partitionKey: The partition key for BigQuery partitioned table.
    tablePrefix: Required. The prefix of the BigQuery tables to which the
      analysis results will be written. Tables will be created based on this
      table_prefix if not exist: * _analysis table will contain export
      operation's metadata. * _analysis_result will contain all the
      IamPolicyAnalysisResult. When [partition_key] is specified, both tables
      will be partitioned based on the [partition_key].
    writeDisposition: Optional. Specifies the action that occurs if the
      destination table or partition already exists. The following values are
      supported: * WRITE_TRUNCATE: If the table or partition already exists,
      BigQuery overwrites the entire table or all the partitions data. *
      WRITE_APPEND: If the table or partition already exists, BigQuery appends
      the data to the table or the latest partition. * WRITE_EMPTY: If the
      table already exists and contains data, an error is returned. The
      default value is WRITE_APPEND. Each action is atomic and only occurs if
      BigQuery is able to complete the job successfully. Details are at
      https://cloud.google.com/bigquery/docs/loading-data-
      local#appending_to_or_overwriting_a_table_using_a_local_file.
  """

    class PartitionKeyValueValuesEnum(_messages.Enum):
        """The partition key for BigQuery partitioned table.

    Values:
      PARTITION_KEY_UNSPECIFIED: Unspecified partition key. Tables won't be
        partitioned using this option.
      REQUEST_TIME: The time when the request is received. If specified as
        partition key, the result table(s) is partitoned by the RequestTime
        column, an additional timestamp column representing when the request
        was received.
    """
        PARTITION_KEY_UNSPECIFIED = 0
        REQUEST_TIME = 1
    dataset = _messages.StringField(1)
    partitionKey = _messages.EnumField('PartitionKeyValueValuesEnum', 2)
    tablePrefix = _messages.StringField(3)
    writeDisposition = _messages.StringField(4)