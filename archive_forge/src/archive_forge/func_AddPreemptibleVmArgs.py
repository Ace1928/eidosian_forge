from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import functools
import ipaddress
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import containers_utils
from googlecloudsdk.api_lib.compute import csek_utils
from googlecloudsdk.api_lib.compute import disks_util
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.compute import kms_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute.zones import service as zones_service
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources as core_resources
import six
def AddPreemptibleVmArgs(parser, is_update=False):
    """Set preemptible scheduling property for instances.

  For set_scheduling operation, in addition we are providing no-preemptible flag
  in use case when user wants to disable this property.

  Args:
     parser: ArgumentParser, parser to which flags will be added.
     is_update: Bool. If True, flags are intended for set-scheduling operation.
  """
    help_text = '      If provided, instances will be preemptible and time-limited. Instances\n      might be preempted to free up resources for standard VM instances,\n      and will only be able to run for a limited amount of time. Preemptible\n      instances can not be restarted and will not migrate.\n      '
    if is_update:
        parser.add_argument('--preemptible', action=arg_parsers.StoreTrueFalseAction, help=help_text)
    else:
        parser.add_argument('--preemptible', action='store_true', default=False, help=help_text)