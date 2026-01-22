from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute.networks.subnets import flags
import six
def CreateSecondaryRanges(client, secondary_range, secondary_range_with_reserved_internal_range):
    """Creates all secondary ranges of a subnet."""
    secondary_ranges = []
    range_name_to_cidr = {}
    range_name_to_reserved_internal_range = {}
    range_names = set()
    if secondary_range:
        for secondary_range in secondary_range:
            for range_name, ip_cidr_range in six.iteritems(secondary_range):
                range_name_to_cidr[range_name] = ip_cidr_range
                range_names.add(range_name)
    if secondary_range_with_reserved_internal_range:
        for secondary_range in secondary_range_with_reserved_internal_range:
            for range_name, internal_range in six.iteritems(secondary_range):
                range_name_to_reserved_internal_range[range_name] = internal_range
                range_names.add(range_name)
    for range_name in sorted(range_names):
        secondary_ranges.append(_CreateSecondaryRange(client, range_name, range_name_to_cidr.get(range_name), range_name_to_reserved_internal_range.get(range_name)))
    return secondary_ranges