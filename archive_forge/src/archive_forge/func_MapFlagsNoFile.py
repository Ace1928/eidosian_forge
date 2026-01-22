import enum
import os
import re
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.container import kubeconfig
from googlecloudsdk.api_lib.run import container_resource
from googlecloudsdk.api_lib.run import global_methods
from googlecloudsdk.api_lib.run import k8s_object
from googlecloudsdk.api_lib.run import revision
from googlecloudsdk.api_lib.run import service
from googlecloudsdk.api_lib.run import traffic
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.services import exceptions as services_exceptions
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.functions.v2.deploy import env_vars_util
from googlecloudsdk.command_lib.run import config_changes
from googlecloudsdk.command_lib.run import exceptions as serverless_exceptions
from googlecloudsdk.command_lib.run import platforms
from googlecloudsdk.command_lib.run import pretty_print
from googlecloudsdk.command_lib.run import resource_args
from googlecloudsdk.command_lib.run import secrets_mapping
from googlecloudsdk.command_lib.run import volumes
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
def MapFlagsNoFile(flag_name, group_help='', long_name=None, key_type=None, value_type=None, key_metavar='KEY', value_metavar='VALUE'):
    """Create an argument group like map_util.AddUpdateMapFlags but without the file one.

  Args:
    flag_name: The name for the property to be used in flag names
    group_help: Help text for the group of flags
    long_name: The name for the property to be used in help text
    key_type: A function to apply to map keys.
    value_type: A function to apply to map values.
    key_metavar: Metavariable to list for the key.
    value_metavar: Metavariable to list for the value.

  Returns:
    A mutually exclusive group for the map flags.
  """
    if not long_name:
        long_name = flag_name
    group = base.ArgumentGroup(mutex=True, help=group_help)
    update_remove_group = base.ArgumentGroup(help='Only --update-{0} and --remove-{0} can be used together. If both are specified, --remove-{0} will be applied first.'.format(flag_name))
    update_remove_group.AddArgument(map_util.MapUpdateFlag(flag_name, long_name, key_type=key_type, value_type=value_type, key_metavar=key_metavar, value_metavar=value_metavar))
    update_remove_group.AddArgument(map_util.MapRemoveFlag(flag_name, long_name, key_type=key_type, key_metavar=key_metavar))
    group.AddArgument(update_remove_group)
    group.AddArgument(map_util.MapClearFlag(flag_name, long_name))
    group.AddArgument(map_util.MapSetFlag(flag_name, long_name, key_type=key_type, value_type=value_type, key_metavar=key_metavar, value_metavar=value_metavar))
    return group