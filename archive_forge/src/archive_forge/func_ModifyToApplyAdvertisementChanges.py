from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
from apitools.base.py import encoding
from googlecloudsdk.api_lib.edge_cloud.networking import utils
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core import exceptions as core_exceptions
import six
def ModifyToApplyAdvertisementChanges(self, args, existing):
    """Create a router based on `existing` with the routes change."""

    def cidrset(cidr_strs):
        return set((ipaddress.ip_network(cidrstr) for cidrstr in cidr_strs))

    def sorted_strings(cidrs):
        return [six.text_type(cidr) for cidr in sorted(cidrs)]
    advertisements = cidrset(existing.routeAdvertisements)
    replacement = encoding.CopyProtoMessage(existing)
    if args.add_advertisement_ranges:
        to_add = set(args.add_advertisement_ranges)
        already_present = sorted_strings(advertisements & to_add)
        if already_present:
            raise core_exceptions.Error('attempting to add routes that are already present: {}'.format(', '.join(already_present)))
        advertisements |= to_add
    elif args.remove_advertisement_ranges:
        to_rm = cidrset(args.remove_advertisement_ranges)
        already_missing = sorted_strings(to_rm - advertisements)
        if already_missing:
            raise core_exceptions.Error('attempting to remove routes that are not present: {}'.format(', '.join(already_missing)))
        advertisements -= to_rm
    elif args.set_advertisement_ranges:
        advertisements = cidrset(args.set_advertisement_ranges)
    else:
        raise parser_errors.ArgumentException('Missing --add-advertisement-ranges, --remove-advertisement-ranges, or --set-advertisement-ranges')
    replacement.routeAdvertisements = list(map(str, sorted(advertisements)))
    return replacement