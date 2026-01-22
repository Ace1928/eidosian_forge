from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import name_generator
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.addresses import flags
import ipaddr
from six.moves import zip  # pylint: disable=redefined-builtin
def _GetNamesAndAddresses(self, args):
    """Returns names and addresses provided in args."""
    if not args.addresses and (not args.name):
        raise exceptions.MinimumArgumentException(['NAME', '--address'], 'At least one name or address must be provided.')
    if args.name:
        names = args.name
    else:
        names = [name_generator.GenerateRandomName() for _ in args.addresses]
    if args.addresses:
        addresses = args.addresses
    else:
        addresses = [None] * len(args.name)
    if len(addresses) != len(names):
        raise exceptions.BadArgumentException('--addresses', 'If providing both, you must specify the same number of names as addresses.')
    return (names, addresses)