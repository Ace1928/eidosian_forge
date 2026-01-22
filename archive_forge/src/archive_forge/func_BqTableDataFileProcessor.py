from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import datetime
import time
import uuid
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.args import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import times
def BqTableDataFileProcessor(file_arg):
    """Convert Input JSON file into TableSchema message."""
    data_insert_request_type = GetApiMessage('TableDataInsertAllRequest')
    insert_row_type = data_insert_request_type.RowsValueListEntry
    data_row_type = GetApiMessage('JsonObject')
    try:
        data_json = yaml.load(file_arg)
        if not data_json or not isinstance(data_json, list):
            raise TableDataFileError('Error parsing data file: no data records defined in file')
        rows = []
        for row in data_json:
            rows.append(insert_row_type(json=encoding.DictToMessage(row, data_row_type)))
        return rows
    except yaml.YAMLParseError as ype:
        raise TableDataFileError('Error parsing data file [{}]'.format(ype))