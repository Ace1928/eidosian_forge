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
def ValidateNicFlags(args):
    """Validates flags specifying network interface cards.

  Throws exceptions or print warning if incompatible nic args are specified.

  Args:
    args: parsed command line arguments.

  Raises:
    InvalidArgumentException: when it finds --network-interface that has both
                              address, and no-address keys.
    ConflictingArgumentsException: when it finds --network-interface and at
                                   least one of --address, --network,
                                   --private_network_ip, or --subnet.
  """
    network_interface = getattr(args, 'network_interface', None)
    network_interface_from_file = getattr(args, 'network_interface_from_file', None)
    network_interface_from_json = getattr(args, 'network_interface_from_json_string', None)
    if network_interface is None and network_interface_from_file is None and (network_interface_from_json is None):
        return
    elif network_interface is not None:
        for ni in network_interface:
            if 'address' in ni and 'no-address' in ni:
                raise exceptions.InvalidArgumentException('--network-interface', 'specifies both address and no-address for one interface')
    conflicting_args = ['address', 'network', 'private_network_ip', 'subnet']
    conflicting_args_present = [arg for arg in conflicting_args if getattr(args, arg, None)]
    conflicting_args = ['--{0}'.format(arg.replace('_', '-')) for arg in conflicting_args_present]
    warning_args = ['network_tier', 'stack_type', 'ipv6_network_tier', 'ipv6_public_ptr_domain', 'internal_ipv6_address', 'internal_ipv6_prefix_length', 'ipv6_address', 'ipv6_prefix_length', 'external_ipv6_address', 'external_ipv6_prefix_length']
    warning_args_present = [arg for arg in warning_args if getattr(args, arg, None)]
    warning_args = ['--{0}'.format(arg.replace('_', '-')) for arg in warning_args_present]
    if not conflicting_args_present and (not warning_args_present):
        return
    if conflicting_args_present:
        if network_interface is not None:
            raise exceptions.ConflictingArgumentsException('--network-interface', 'all of the following: ' + ', '.join(conflicting_args))
        elif network_interface_from_file is not None:
            raise exceptions.ConflictingArgumentsException('--network-interface-from-file', 'all of the following: ' + ', '.join(conflicting_args))
        else:
            raise exceptions.ConflictingArgumentsException('--network-interface-from-json-string', 'all of the following: ' + ', '.join(conflicting_args))
    if warning_args_present:
        if network_interface is not None:
            nic_arg_name = '--network-interface'
        elif network_interface_from_file is not None:
            nic_arg_name = '--network-interface-from-file'
        else:
            nic_arg_name = '--network-interface-from-json-string'
        log.status.write(f'When {nic_arg_name} is specified, the following arguments are ignored: ' + ', '.join(warning_args) + '.\n')