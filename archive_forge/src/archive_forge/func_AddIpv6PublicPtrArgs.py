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
def AddIpv6PublicPtrArgs(parser):
    """Adds IPv6 public PTR arguments for ipv6 access configuration."""
    ipv6_public_ptr_args = parser.add_mutually_exclusive_group()
    no_ipv6_public_ptr_help = '        If provided, the default DNS PTR record will replace the existing one\n        for external IPv6 in the IPv6 access configuration. Mutually exclusive\n        with ipv6-public-ptr-domain.\n        '
    ipv6_public_ptr_args.add_argument('--no-ipv6-public-ptr', action='store_true', help=no_ipv6_public_ptr_help)
    ipv6_public_ptr_domain_help = "        Assigns a custom PTR domain for the external IPv6 in the access\n        configuration. Mutually exclusive with no-ipv6-public-ptr. This option\n        can only be specified for the default network interface, ``nic0''.\n        "
    ipv6_public_ptr_args.add_argument('--ipv6-public-ptr-domain', help=ipv6_public_ptr_domain_help)