from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import itertools
import logging
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional
import uuid
from absl import flags
from google.api_core.iam import Policy
from googleapiclient import http as http_request
import inflection
from clients import bigquery_client
from clients import client_dataset
from clients import client_reservation
from clients import table_reader as bq_table_reader
from clients import utils as bq_client_utils
from utils import bq_api_utils
from utils import bq_error
from utils import bq_id_utils
from utils import bq_processor_utils
def Load(self, destination_table_reference, source, schema=None, create_disposition=None, write_disposition=None, field_delimiter=None, skip_leading_rows=None, encoding=None, quote=None, max_bad_records=None, allow_quoted_newlines=None, source_format=None, allow_jagged_rows=None, preserve_ascii_control_characters=None, ignore_unknown_values=None, projection_fields=None, autodetect=None, schema_update_options=None, null_marker=None, time_partitioning=None, clustering=None, destination_encryption_configuration=None, use_avro_logical_types=None, reference_file_schema_uri=None, range_partitioning=None, hive_partitioning_options=None, decimal_target_types=None, json_extension=None, file_set_spec_type=None, thrift_options=None, parquet_options=None, connection_properties=None, copy_files_only: Optional[bool]=None, **kwds):
    """Load the given data into BigQuery.

    The job will execute synchronously if sync=True is provided as an
    argument or if self.sync is true.

    Args:
      destination_table_reference: TableReference to load data into.
      source: String specifying source data to load.
      schema: (default None) Schema of the created table. (Can be left blank for
        append operations.)
      create_disposition: Optional. Specifies the create_disposition for the
        destination_table_reference.
      write_disposition: Optional. Specifies the write_disposition for the
        destination_table_reference.
      field_delimiter: Optional. Specifies the single byte field delimiter.
      skip_leading_rows: Optional. Number of rows of initial data to skip.
      encoding: Optional. Specifies character encoding of the input data. May be
        "UTF-8" or "ISO-8859-1". Defaults to UTF-8 if not specified.
      quote: Optional. Quote character to use. Default is '"'. Note that quoting
        is done on the raw binary data before encoding is applied.
      max_bad_records: Optional. Maximum number of bad records that should be
        ignored before the entire job is aborted. Only supported for CSV and
        NEWLINE_DELIMITED_JSON file formats.
      allow_quoted_newlines: Optional. Whether to allow quoted newlines in CSV
        import data.
      source_format: Optional. Format of source data. May be "CSV",
        "DATASTORE_BACKUP", or "NEWLINE_DELIMITED_JSON".
      allow_jagged_rows: Optional. Whether to allow missing trailing optional
        columns in CSV import data.
      preserve_ascii_control_characters: Optional. Whether to preserve embedded
        Ascii Control characters in CSV import data.
      ignore_unknown_values: Optional. Whether to allow extra, unrecognized
        values in CSV or JSON data.
      projection_fields: Optional. If sourceFormat is set to "DATASTORE_BACKUP",
        indicates which entity properties to load into BigQuery from a Cloud
        Datastore backup.
      autodetect: Optional. If true, then we automatically infer the schema and
        options of the source files if they are CSV or JSON formats.
      schema_update_options: schema update options when appending to the
        destination table or truncating a table partition.
      null_marker: Optional. String that will be interpreted as a NULL value.
      time_partitioning: Optional. Provides time based partitioning
        specification for the destination table.
      clustering: Optional. Provides clustering specification for the
        destination table.
      destination_encryption_configuration: Optional. Allows user to encrypt the
        table created from a load job with Cloud KMS key.
      use_avro_logical_types: Optional. Allows user to override default
        behaviour for Avro logical types. If this is set, Avro fields with
        logical types will be interpreted into their corresponding types (ie.
        TIMESTAMP), instead of only using their raw types (ie. INTEGER).
      reference_file_schema_uri: Optional. Allows user to provide a reference
        file with the reader schema, enabled for the format: AVRO, PARQUET, ORC.
      range_partitioning: Optional. Provides range partitioning specification
        for the destination table.
      hive_partitioning_options: (experimental) Options for configuring hive is
        picked if it is in the specified list and if it supports the precision
        and the scale. STRING supports all precision and scale values. If none
        of the listed types supports the precision and the scale, the type
        supporting the widest range in the specified list is picked, and if a
        value exceeds the supported range when reading the data, an error will
        be returned. This field cannot contain duplicate types. The order of the
      decimal_target_types: (experimental) Defines the list of possible SQL data
        types to which the source decimal values are converted. This list and
        the precision and the scale parameters of the decimal field determine
        the target type. In the order of NUMERIC, BIGNUMERIC, and STRING, a type
        is picked if it is in the specified list and if it supports the
        precision and the scale. STRING supports all precision and scale values.
        If none of the listed types supports the precision and the scale, the
        type supporting the widest range in the specified list is picked, and if
        a value exceeds the supported range when reading the data, an error will
        be returned. This field cannot contain duplicate types. The order of the
        types in this field is ignored. For example, ["BIGNUMERIC", "NUMERIC"]
        is the same as ["NUMERIC", "BIGNUMERIC"] and NUMERIC always takes
        precedence over BIGNUMERIC. Defaults to ["NUMERIC", "STRING"] for ORC
        and ["NUMERIC"] for the other file formats.
      json_extension: (experimental) Specify alternative parsing for JSON source
        format. To load newline-delimited JSON, specify 'GEOJSON'. Only
        applicable if `source_format` is 'NEWLINE_DELIMITED_JSON'.
      file_set_spec_type: (experimental) Set how to discover files for loading.
        Specify 'FILE_SYSTEM_MATCH' (default behavior) to expand source URIs by
        listing files from the underlying object store. Specify
        'NEW_LINE_DELIMITED_MANIFEST' to parse the URIs as new line delimited
        manifest files, where each line contains a URI (No wild-card URIs are
        supported).
      thrift_options: (experimental) Options for configuring Apache Thrift load,
        which is required if `source_format` is 'THRIFT'.
      parquet_options: Options for configuring parquet files load, only
        applicable if `source_format` is 'PARQUET'.
      connection_properties: Optional. ConnectionProperties for load job.
      copy_files_only: Optional. True to configures the load job to only copy
        files to the destination BigLake managed table, without reading file
        content and writing them to new files.
      **kwds: Passed on to self.ExecuteJob.

    Returns:
      The resulting job info.
    """
    bq_id_utils.typecheck(destination_table_reference, bq_id_utils.ApiClientHelper.TableReference)
    load_config = {'destinationTable': dict(destination_table_reference)}
    sources = bq_client_utils.ProcessSources(source)
    if sources[0].startswith(bq_processor_utils.GCS_SCHEME_PREFIX):
        load_config['sourceUris'] = sources
        upload_file = None
    else:
        upload_file = sources[0]
    if schema is not None:
        load_config['schema'] = {'fields': bq_client_utils.ReadSchema(schema)}
    if use_avro_logical_types is not None:
        load_config['useAvroLogicalTypes'] = use_avro_logical_types
    if reference_file_schema_uri is not None:
        load_config['reference_file_schema_uri'] = reference_file_schema_uri
    if file_set_spec_type is not None:
        load_config['fileSetSpecType'] = file_set_spec_type
    if json_extension is not None:
        load_config['jsonExtension'] = json_extension
    if parquet_options is not None:
        load_config['parquetOptions'] = parquet_options
    load_config['decimalTargetTypes'] = decimal_target_types
    if destination_encryption_configuration:
        load_config['destinationEncryptionConfiguration'] = destination_encryption_configuration
    bq_processor_utils.ApplyParameters(load_config, create_disposition=create_disposition, write_disposition=write_disposition, field_delimiter=field_delimiter, skip_leading_rows=skip_leading_rows, encoding=encoding, quote=quote, max_bad_records=max_bad_records, source_format=source_format, allow_quoted_newlines=allow_quoted_newlines, allow_jagged_rows=allow_jagged_rows, preserve_ascii_control_characters=preserve_ascii_control_characters, ignore_unknown_values=ignore_unknown_values, projection_fields=projection_fields, schema_update_options=schema_update_options, null_marker=null_marker, time_partitioning=time_partitioning, clustering=clustering, autodetect=autodetect, range_partitioning=range_partitioning, hive_partitioning_options=hive_partitioning_options, thrift_options=thrift_options, connection_properties=connection_properties, copy_files_only=copy_files_only, parquet_options=parquet_options)
    return self.ExecuteJob(configuration={'load': load_config}, upload_file=upload_file, **kwds)