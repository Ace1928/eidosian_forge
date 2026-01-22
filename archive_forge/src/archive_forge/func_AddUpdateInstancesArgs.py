from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from typing import Any
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddUpdateInstancesArgs(parser):
    """Add args for the update-instances command."""
    instance_selector_group = parser.add_group(required=True, mutex=True)
    instance_selector_group.add_argument('--instances', type=arg_parsers.ArgList(min_length=1), metavar='INSTANCE', required=False, help='Names of instances to update.')
    instance_selector_group.add_argument('--all-instances', required=False, action='store_true', help='Update all instances in the group.')
    AddMinimalActionArg(parser, True, 'none')
    AddMostDisruptiveActionArg(parser, True, 'replace')