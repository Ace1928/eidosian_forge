from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import enum
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import arg_parsers_usage_text as usage_text
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import util as format_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import yaml_command_schema_util as util
import six
def _GetTextFormatOfEmptyValue(self, value):
    if value:
        return value
    if isinstance(value, dict):
        return 'empty map'
    if isinstance(value, list):
        return 'empty list'
    if value is None:
        return 'null'
    return value