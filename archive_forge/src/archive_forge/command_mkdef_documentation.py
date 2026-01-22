from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import sys
from typing import Optional
from absl import flags
from frontend import bigquery_command
from frontend import flags as frontend_flags
from frontend import utils as frontend_utils
Emits a definition in JSON for an external table, such as GCS.

    The output of this command can be redirected to a file and used for the
    external_table_definition flag with the "bq query" and "bq mk" commands.
    It produces a definition with the most commonly used values for options.
    You can modify the output to override option values.

    The <source_uris> argument is a comma-separated list of URIs indicating
    the data referenced by this external table.

    The <schema> argument should be either the name of a JSON file or a text
    schema.

    In the case that the schema is provided in text form, it should be a
    comma-separated list of entries of the form name[:type], where type will
    default to string if not specified.

    In the case that <schema> is a filename, it should be a JSON file
    containing a single array, each entry of which should be an object with
    properties 'name', 'type', and (optionally) 'mode'. For more detail:
    https://cloud.google.com/bigquery/docs/schemas#specifying_a_json_schema_file

    Note: the case of a single-entry schema with no type specified is
    ambiguous; one can use name:string to force interpretation as a
    text schema.

    Usage:
      mkdef <source_uris> [<schema>]

    Examples:
      bq mkdef 'gs://bucket/file.csv' field1:integer,field2:string

    Arguments:
      source_uris: Comma-separated list of URIs.
      schema: Either a text schema or JSON file, as above.
    