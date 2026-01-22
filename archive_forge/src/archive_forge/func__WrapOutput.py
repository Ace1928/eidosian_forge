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
def _WrapOutput(self, output_list):
    """Wraps field AdditionalProperties in apitools message if needed.

    Args:
      output_list: list of apitools AdditionalProperties messages.

    Returns:
      apitools message instance.
    """
    if self._is_list_field:
        return output_list
    message = self.field.type()
    arg_utils.SetFieldInMessage(message, arg_utils.ADDITIONAL_PROPS, output_list)
    return message