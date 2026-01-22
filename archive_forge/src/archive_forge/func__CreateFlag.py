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
def _CreateFlag(self, arg_name, flag_prefix=None, flag_type=None, action=None, metavar=None, help_text=None):
    """Creates a flag.

    Args:
      arg_name: str, root name of the arg
      flag_prefix: Prefix | None, prefix for the flag name
      flag_type: func, type that flag is used to convert user input
      action: str, flag action
      metavar: str, user specified metavar for flag
      help_text: str, flag help text

    Returns:
      base.Argument with correct params
    """
    flag_name = arg_utils.GetFlagName(arg_name, flag_prefix and flag_prefix.value)
    arg = base.Argument(flag_name, action=action, help=help_text)
    if action == 'store_true':
        return arg
    arg.kwargs['type'] = flag_type
    if (flag_metavar := arg_utils.GetMetavar(metavar, flag_type, flag_name)):
        arg.kwargs['metavar'] = flag_metavar
    return arg