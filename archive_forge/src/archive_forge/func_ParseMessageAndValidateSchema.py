from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import uuid
from apitools.base.py import encoding as api_encoding
from googlecloudsdk.api_lib.dataproc import exceptions
from googlecloudsdk.api_lib.datastream import camel_case_utils
from googlecloudsdk.api_lib.datastream import exceptions as ds_exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.export import util as export_util
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
import six
def ParseMessageAndValidateSchema(config_file_path, schema_name, message_type):
    """Parses a config message and validates it's schema."""
    schema_path = export_util.GetSchemaPath(_DEFAULT_API_NAME, _DEFAULT_API_VERSION, schema_name, for_help=False)
    data = console_io.ReadFromFileOrStdin(config_file_path, binary=False)
    parsed_yaml = yaml.load(data)
    message = CreateMessageWithCamelCaseConversion(message_type=message_type, parsed_yaml=parsed_yaml, schema_path=schema_path)
    return message