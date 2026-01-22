import itertools as _itertools
import sys as _sys
from netaddr.ip import IPNetwork, IPAddress, IPRange, cidr_merge, cidr_exclude, iprange_to_cidrs
def _iter_merged_ranges(sorted_ranges):
    """Iterate over sorted_ranges, merging where possible

    Sorted ranges must be a sorted iterable of (version, first, last) tuples.
    Merging occurs for pairs like [(4, 10, 42), (4, 43, 100)] which is merged
    into (4, 10, 100), and leads to return value
    ( IPAddress(10, 4), IPAddress(100, 4) ), which is suitable input for the
    iprange_to_cidrs function.
    """
    if not sorted_ranges:
        return
    current_version, current_start, current_stop = sorted_ranges[0]
    for next_version, next_start, next_stop in sorted_ranges[1:]:
        if next_start == current_stop + 1 and next_version == current_version:
            current_stop = next_stop
            continue
        yield (IPAddress(current_start, current_version), IPAddress(current_stop, current_version))
        current_start = next_start
        current_stop = next_stop
        current_version = next_version
    yield (IPAddress(current_start, current_version), IPAddress(current_stop, current_version))