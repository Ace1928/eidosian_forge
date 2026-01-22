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
def AddStackTypeArgs(parser, support_ipv6_only=False):
    """Adds stack type arguments for instance."""
    choices = {'IPV4_ONLY': 'The network interface will be assigned IPv4 addresses', 'IPV4_IPV6': 'The network interface can have both IPv4 and IPv6 addresses'}
    if support_ipv6_only:
        choices['IPV6_ONLY'] = 'The network interface will be assigned IPv6 addresses'
    parser.add_argument('--stack-type', choices=choices, type=arg_utils.ChoiceToEnumName, help='Specifies whether IPv6 is enabled on the default network interface. If not specified, IPV4_ONLY will be used.')