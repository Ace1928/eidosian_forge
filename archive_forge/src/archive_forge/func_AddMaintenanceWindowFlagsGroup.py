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
def AddMaintenanceWindowFlagsGroup(create_type_group):
    """Adds flag group for maintenance window.

  Args:
    create_type_group: argument group, the group to which flags should be added.
  """
    group = create_type_group.add_group(MAINTENANCE_WINDOW_FLAG_GROUP_DESCRIPTION)
    MAINTENANCE_WINDOW_START_FLAG.AddToParser(group)
    MAINTENANCE_WINDOW_END_FLAG.AddToParser(group)
    MAINTENANCE_WINDOW_RECURRENCE_FLAG.AddToParser(group)