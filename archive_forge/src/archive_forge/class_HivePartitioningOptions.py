from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HivePartitioningOptions(_messages.Message):
    """Options for configuring hive partitioning detect.

  Fields:
    fields: Output only. For permanent external tables, this field is
      populated with the hive partition keys in the order they were inferred.
      The types of the partition keys can be deduced by checking the table
      schema (which will include the partition keys). Not every API will
      populate this field in the output. For example, Tables.Get will populate
      it, but Tables.List will not contain this field.
    mode: Optional. When set, what mode of hive partitioning to use when
      reading data. The following modes are supported: * AUTO: automatically
      infer partition key name(s) and type(s). * STRINGS: automatically infer
      partition key name(s). All types are strings. * CUSTOM: partition key
      schema is encoded in the source URI prefix. Not all storage formats
      support hive partitioning. Requesting hive partitioning on an
      unsupported format will lead to an error. Currently supported formats
      are: JSON, CSV, ORC, Avro and Parquet.
    requirePartitionFilter: Optional. If set to true, queries over this table
      require a partition filter that can be used for partition elimination to
      be specified. Note that this field should only be true when creating a
      permanent external table or querying a temporary external table. Hive-
      partitioned loads with require_partition_filter explicitly set to true
      will fail.
    sourceUriPrefix: Optional. When hive partition detection is requested, a
      common prefix for all source uris must be required. The prefix must end
      immediately before the partition key encoding begins. For example,
      consider files following this data layout:
      gs://bucket/path_to_table/dt=2019-06-01/country=USA/id=7/file.avro
      gs://bucket/path_to_table/dt=2019-05-31/country=CA/id=3/file.avro When
      hive partitioning is requested with either AUTO or STRINGS detection,
      the common prefix can be either of gs://bucket/path_to_table or
      gs://bucket/path_to_table/. CUSTOM detection requires encoding the
      partitioning schema immediately after the common prefix. For CUSTOM, any
      of * gs://bucket/path_to_table/{dt:DATE}/{country:STRING}/{id:INTEGER} *
      gs://bucket/path_to_table/{dt:STRING}/{country:STRING}/{id:INTEGER} *
      gs://bucket/path_to_table/{dt:DATE}/{country:STRING}/{id:STRING} would
      all be valid source URI prefixes.
  """
    fields = _messages.StringField(1, repeated=True)
    mode = _messages.StringField(2)
    requirePartitionFilter = _messages.BooleanField(3, default=False)
    sourceUriPrefix = _messages.StringField(4)