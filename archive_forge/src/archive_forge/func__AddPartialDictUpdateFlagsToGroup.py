from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import ipaddress
import re
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.composer import parsers
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import properties
import six
def _AddPartialDictUpdateFlagsToGroup(update_type_group, clear_flag, remove_flag, update_flag, group_help_text=None):
    """Adds flags related to a partial update of arg represented by a dictionary.

  Args:
    update_type_group: argument group, the group to which flags should be added.
    clear_flag: flag, the flag to clear dictionary.
    remove_flag: flag, the flag to remove values from dictionary.
    update_flag: flag, the flag to add or update values in dictionary.
    group_help_text: (optional) str, the help info to apply to the created
      argument group. If not provided, then no help text will be applied to
      group.
  """
    group = update_type_group.add_argument_group(help=group_help_text)
    remove_group = group.add_mutually_exclusive_group(help=GENERAL_REMOVAL_FLAG_GROUP_DESCRIPTION)
    clear_flag.AddToParser(remove_group)
    remove_flag.AddToParser(remove_group)
    update_flag.AddToParser(group)