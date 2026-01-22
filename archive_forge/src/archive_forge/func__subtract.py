import itertools as _itertools
import sys as _sys
from netaddr.ip import IPNetwork, IPAddress, IPRange, cidr_merge, cidr_exclude, iprange_to_cidrs
def _subtract(supernet, subnets, subnet_idx, ranges):
    """Calculate IPSet([supernet]) - IPSet(subnets).

    Assumptions: subnets is sorted, subnet_idx points to the first
    element in subnets that is a subnet of supernet.

    Results are appended to the ranges parameter as tuples of in format
    (version, first, last). Return value is the first subnet_idx that
    does not point to a subnet of supernet (or len(subnets) if all
    subsequents items are a subnet of supernet).
    """
    version = supernet._module.version
    subnet = subnets[subnet_idx]
    if subnet.first > supernet.first:
        ranges.append((version, supernet.first, subnet.first - 1))
    subnet_idx += 1
    prev_subnet = subnet
    while subnet_idx < len(subnets):
        cur_subnet = subnets[subnet_idx]
        if cur_subnet not in supernet:
            break
        if prev_subnet.last + 1 == cur_subnet.first:
            pass
        else:
            ranges.append((version, prev_subnet.last + 1, cur_subnet.first - 1))
        subnet_idx += 1
        prev_subnet = cur_subnet
    first = prev_subnet.last + 1
    last = supernet.last
    if first <= last:
        ranges.append((version, first, last))
    return subnet_idx