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
def AddCustomMachineTypeArgs(parser):
    """Adds arguments related to custom machine types for instances."""
    custom_group = parser.add_group(help='Custom machine type extensions.')
    custom_group.add_argument('--custom-cpu', type=NonEmptyString('--custom-cpu'), required=True, help='      A whole number value specifying the number of cores that are needed in\n      the custom machine type.\n\n      For some machine types, shared-core values can also be used. For\n      example, for E2 machine types, you can specify `micro`, `small`, or\n      `medium`.\n      ')
    custom_group.add_argument('--custom-memory', type=arg_parsers.BinarySize(), required=True, help='      A whole number value indicating how much memory is desired in the custom\n      machine type. A size unit should be provided (eg. 3072MB or 9GB) - if no\n      units are specified, GB is assumed.\n      ')
    custom_group.add_argument('--custom-extensions', action='store_true', help='Use the extended custom machine type.')
    custom_group.add_argument('--custom-vm-type', help='\n      Specifies a custom machine type. The default is `n1`. For more information about custom machine types, see:\n      https://cloud.google.com/compute/docs/general-purpose-machines#custom_machine_types\n      ')