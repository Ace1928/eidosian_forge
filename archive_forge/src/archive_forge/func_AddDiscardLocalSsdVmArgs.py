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
def AddDiscardLocalSsdVmArgs(parser, is_update=False):
    """Set arguments for specifing discard-local-ssds-at-termination-timestamp flag."""
    discard_local_ssds_at_termination_timestamp_help_text = "        Required and only allowed for VMs that have one or more local SSDs,\n        use --termination-action=STOP (default), and use either\n        --max-run-duration or --termination-time. This flag indicates whether\n        you want Compute Engine to discard (true) or preserve (false) local SSD\n        data when the VM's terminationTimestamp is reached.\n\n        If set to false, Compute Engine will preserve local SSD data by\n        including the --discard-local-ssd=false flag in the automatic\n        termination command. The --discard-local-ssd=false flag preserves local\n        SSD data by migrating it to persistent storage until you rerun the VM.\n        Importantly, preserving local SSD data incurs costs and is subject to\n        restrictions. For more information, see https://cloud.google.com/compute/docs/disks/local-ssd#stop_instance.\n      "
    if is_update:
        discard_local_ssds_at_termination_timestamp_group = parser.add_group('Discard Local SSDs At Termination Timestamp', mutex=True)
        discard_local_ssds_at_termination_timestamp_group.add_argument('--clear-discard-local-ssds-at-termination-timestamp', action='store_true', help='        Removes the discard-local-ssds-at-termination-timestamp field from the scheduling options.\n        ')
        discard_local_ssds_at_termination_timestamp_group.add_argument('--discard-local-ssds-at-termination-timestamp', type=arg_parsers.ArgBoolean(), help=discard_local_ssds_at_termination_timestamp_help_text)
    else:
        parser.add_argument('--discard-local-ssds-at-termination-timestamp', type=arg_parsers.ArgBoolean(), help=discard_local_ssds_at_termination_timestamp_help_text)