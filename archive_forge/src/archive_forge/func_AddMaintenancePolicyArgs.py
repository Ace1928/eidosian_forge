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
def AddMaintenancePolicyArgs(parser, deprecate=False):
    """Adds maintenance behavior related args."""
    help_text = '  Specifies the behavior of the VMs when their host machines undergo\n  maintenance. The default is MIGRATE.\n  For more information, see\n  https://cloud.google.com/compute/docs/instances/host-maintenance-options.\n  '
    flag_type = lambda x: x.upper()
    action = None
    if deprecate:
        parser = parser.add_mutually_exclusive_group('Maintenance Behavior.')
        parser.add_argument('--on-host-maintenance', dest='maintenance_policy', choices=MIGRATION_OPTIONS, type=flag_type, help=help_text)
        action = actions.DeprecationAction('--maintenance-policy', warn='The {flag_name} flag is now deprecated. Please use `--on-host-maintenance` instead')
    parser.add_argument('--maintenance-policy', action=action, choices=MIGRATION_OPTIONS, type=flag_type, help=help_text)