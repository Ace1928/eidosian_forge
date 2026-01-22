import itertools as _itertools
import sys as _sys
from netaddr.ip import IPNetwork, IPAddress, IPRange, cidr_merge, cidr_exclude, iprange_to_cidrs
class IPSet(object):
    """
    Represents an unordered collection (set) of unique IP addresses and
    subnets.

    """
    __slots__ = ('_cidrs', '__weakref__')

    def __init__(self, iterable=None, flags=0):
        """
        Constructor.

        :param iterable: (optional) an iterable containing IP addresses,
            subnets or ranges.

        :param flags: decides which rules are applied to the interpretation
            of the addr value. See the :class:`IPAddress` documentation
            for supported constant values.

        """
        if isinstance(iterable, IPNetwork):
            self._cidrs = {iterable.cidr: True}
        elif isinstance(iterable, IPRange):
            self._cidrs = dict.fromkeys(iprange_to_cidrs(iterable[0], iterable[-1]), True)
        elif isinstance(iterable, IPSet):
            self._cidrs = dict.fromkeys(iterable.iter_cidrs(), True)
        else:
            self._cidrs = {}
            if iterable is not None:
                mergeable = []
                for addr in iterable:
                    if isinstance(addr, int):
                        addr = IPAddress(addr, flags=flags)
                    mergeable.append(addr)
                for cidr in cidr_merge(mergeable):
                    self._cidrs[cidr] = True

    def __getstate__(self):
        """:return: Pickled state of an ``IPSet`` object."""
        return tuple([cidr.__getstate__() for cidr in self._cidrs])

    def __setstate__(self, state):
        """
        :param state: data used to unpickle a pickled ``IPSet`` object.

        """
        self._cidrs = dict.fromkeys((IPNetwork((value, prefixlen), version=version) for value, prefixlen, version in state), True)

    def _compact_single_network(self, added_network):
        """
        Same as compact(), but assume that added_network is the only change and
        that this IPSet was properly compacted before added_network was added.
        This allows to perform compaction much faster. added_network must
        already be present in self._cidrs.
        """
        added_first = added_network.first
        added_last = added_network.last
        added_version = added_network.version
        if added_network._prefixlen == added_network._module.width:
            for potential_supernet in added_network.supernet():
                if potential_supernet in self._cidrs:
                    del self._cidrs[added_network]
                    return
        else:
            to_remove = []
            for cidr in self._cidrs:
                if cidr._module.version != added_version or cidr == added_network:
                    continue
                first = cidr.first
                last = cidr.last
                if first >= added_first and last <= added_last:
                    to_remove.append(cidr)
                elif first <= added_first and last >= added_last:
                    del self._cidrs[added_network]
                    assert not to_remove
                    return
            for item in to_remove:
                del self._cidrs[item]
        shift_width = added_network._module.width - added_network.prefixlen
        while added_network.prefixlen != 0:
            the_bit = added_network._value >> shift_width & 1
            if the_bit:
                candidate = added_network.previous()
            else:
                candidate = added_network.next()
            if candidate not in self._cidrs:
                return
            del self._cidrs[candidate]
            del self._cidrs[added_network]
            added_network.prefixlen -= 1
            shift_width += 1
            added_network._value = added_network._value >> shift_width << shift_width
            self._cidrs[added_network] = True

    def compact(self):
        """
        Compact internal list of `IPNetwork` objects using a CIDR merge.
        """
        cidrs = cidr_merge(self._cidrs)
        self._cidrs = dict.fromkeys(cidrs, True)

    def __hash__(self):
        """
        Raises ``TypeError`` if this method is called.

        .. note:: IPSet objects are not hashable and cannot be used as             dictionary keys or as members of other sets.         """
        raise TypeError('IP sets are unhashable!')

    def __contains__(self, ip):
        """
        :param ip: An IP address or subnet.

        :return: ``True`` if IP address or subnet is a member of this IP set.
        """
        supernet = IPNetwork(ip)
        if supernet in self._cidrs:
            return True
        while supernet._prefixlen:
            supernet._prefixlen -= 1
            if supernet in self._cidrs:
                return True
        return False

    def __bool__(self):
        """Return True if IPSet contains at least one IP, else False"""
        return bool(self._cidrs)

    def __iter__(self):
        """
        :return: an iterator over the IP addresses within this IP set.
        """
        return _itertools.chain(*sorted(self._cidrs))

    def iter_cidrs(self):
        """
        :return: an iterator over individual IP subnets within this IP set.
        """
        return sorted(self._cidrs)

    def add(self, addr, flags=0):
        """
        Adds an IP address or subnet or IPRange to this IP set. Has no effect if
        it is already present.

        Note that where possible the IP address or subnet is merged with other
        members of the set to form more concise CIDR blocks.

        :param addr: An IP address or subnet in either string or object form, or
            an IPRange object.

        :param flags: decides which rules are applied to the interpretation
            of the addr value. See the :class:`IPAddress` documentation
            for supported constant values.

        """
        if isinstance(addr, IPRange):
            new_cidrs = dict.fromkeys(iprange_to_cidrs(addr[0], addr[-1]), True)
            self._cidrs.update(new_cidrs)
            self.compact()
            return
        if isinstance(addr, IPNetwork):
            addr = addr.cidr
        elif isinstance(addr, int):
            addr = IPNetwork(IPAddress(addr, flags=flags))
        else:
            addr = IPNetwork(addr)
        self._cidrs[addr] = True
        self._compact_single_network(addr)

    def remove(self, addr, flags=0):
        """
        Removes an IP address or subnet or IPRange from this IP set. Does
        nothing if it is not already a member.

        Note that this method behaves more like discard() found in regular
        Python sets because it doesn't raise KeyError exceptions if the
        IP address or subnet is question does not exist. It doesn't make sense
        to fully emulate that behaviour here as IP sets contain groups of
        individual IP addresses as individual set members using IPNetwork
        objects.

        :param addr: An IP address or subnet, or an IPRange.

        :param flags: decides which rules are applied to the interpretation
            of the addr value. See the :class:`IPAddress` documentation
            for supported constant values.

        """
        if isinstance(addr, IPRange):
            cidrs = iprange_to_cidrs(addr[0], addr[-1])
            for cidr in cidrs:
                self.remove(cidr)
            return
        if isinstance(addr, int):
            addr = IPAddress(addr, flags=flags)
        else:
            addr = IPNetwork(addr)
        self.add(addr)
        remainder = None
        matching_cidr = None
        for cidr in self._cidrs:
            if addr in cidr:
                remainder = cidr_exclude(cidr, addr)
                matching_cidr = cidr
                break
        if remainder is not None:
            del self._cidrs[matching_cidr]
            for cidr in remainder:
                self._cidrs[cidr] = True

    def pop(self):
        """
        Removes and returns an arbitrary IP address or subnet from this IP
        set.

        :return: An IP address or subnet.
        """
        return self._cidrs.popitem()[0]

    def isdisjoint(self, other):
        """
        :param other: an IP set.

        :return: ``True`` if this IP set has no elements (IP addresses
            or subnets) in common with other. Intersection *must* be an
            empty set.
        """
        result = self.intersection(other)
        return not result

    def copy(self):
        """:return: a shallow copy of this IP set."""
        obj_copy = self.__class__()
        obj_copy._cidrs.update(self._cidrs)
        return obj_copy

    def update(self, iterable, flags=0):
        """
        Update the contents of this IP set with the union of itself and
        other IP set.

        :param iterable: an iterable containing IP addresses, subnets or ranges.

        :param flags: decides which rules are applied to the interpretation
            of the addr value. See the :class:`IPAddress` documentation
            for supported constant values.

        """
        if isinstance(iterable, IPSet):
            self._cidrs = dict.fromkeys((ip for ip in cidr_merge(_itertools.chain(self._cidrs.keys(), iterable._cidrs.keys()))), True)
            return
        elif isinstance(iterable, (IPNetwork, IPRange)):
            self.add(iterable)
            return
        if not hasattr(iterable, '__iter__'):
            raise TypeError('an iterable was expected!')
        mergeable = []
        for addr in iterable:
            if isinstance(addr, int):
                addr = IPAddress(addr, flags=flags)
            mergeable.append(addr)
        for cidr in cidr_merge(_itertools.chain(self._cidrs.keys(), mergeable)):
            self._cidrs[cidr] = True
        self.compact()

    def clear(self):
        """Remove all IP addresses and subnets from this IP set."""
        self._cidrs = {}

    def __eq__(self, other):
        """
        :param other: an IP set

        :return: ``True`` if this IP set is equivalent to the ``other`` IP set,
            ``False`` otherwise.
        """
        try:
            return self._cidrs == other._cidrs
        except AttributeError:
            return NotImplemented

    def __ne__(self, other):
        """
        :param other: an IP set

        :return: ``False`` if this IP set is equivalent to the ``other`` IP set,
            ``True`` otherwise.
        """
        try:
            return self._cidrs != other._cidrs
        except AttributeError:
            return NotImplemented

    def __lt__(self, other):
        """
        :param other: an IP set

        :return: ``True`` if this IP set is less than the ``other`` IP set,
            ``False`` otherwise.
        """
        if not hasattr(other, '_cidrs'):
            return NotImplemented
        return self.size < other.size and self.issubset(other)

    def issubset(self, other):
        """
        :param other: an IP set.

        :return: ``True`` if every IP address and subnet in this IP set
            is found within ``other``.
        """
        for cidr in self._cidrs:
            if cidr not in other:
                return False
        return True
    __le__ = issubset

    def __gt__(self, other):
        """
        :param other: an IP set.

        :return: ``True`` if this IP set is greater than the ``other`` IP set,
            ``False`` otherwise.
        """
        if not hasattr(other, '_cidrs'):
            return NotImplemented
        return self.size > other.size and self.issuperset(other)

    def issuperset(self, other):
        """
        :param other: an IP set.

        :return: ``True`` if every IP address and subnet in other IP set
            is found within this one.
        """
        if not hasattr(other, '_cidrs'):
            return NotImplemented
        for cidr in other._cidrs:
            if cidr not in self:
                return False
        return True
    __ge__ = issuperset

    def union(self, other):
        """
        :param other: an IP set.

        :return: the union of this IP set and another as a new IP set
            (combines IP addresses and subnets from both sets).
        """
        ip_set = self.copy()
        ip_set.update(other)
        return ip_set
    __or__ = union

    def intersection(self, other):
        """
        :param other: an IP set.

        :return: the intersection of this IP set and another as a new IP set.
            (IP addresses and subnets common to both sets).
        """
        result_cidrs = {}
        own_nets = sorted(self._cidrs)
        other_nets = sorted(other._cidrs)
        own_idx = 0
        other_idx = 0
        own_len = len(own_nets)
        other_len = len(other_nets)
        while own_idx < own_len and other_idx < other_len:
            own_cur = own_nets[own_idx]
            other_cur = other_nets[other_idx]
            if own_cur == other_cur:
                result_cidrs[own_cur] = True
                own_idx += 1
                other_idx += 1
            elif own_cur in other_cur:
                result_cidrs[own_cur] = True
                own_idx += 1
            elif other_cur in own_cur:
                result_cidrs[other_cur] = True
                other_idx += 1
            elif own_cur < other_cur:
                own_idx += 1
            else:
                other_idx += 1
        result = IPSet()
        result._cidrs = result_cidrs
        return result
    __and__ = intersection

    def symmetric_difference(self, other):
        """
        :param other: an IP set.

        :return: the symmetric difference of this IP set and another as a new
            IP set (all IP addresses and subnets that are in exactly one
            of the sets).
        """
        result_ranges = []
        own_nets = sorted(self._cidrs)
        other_nets = sorted(other._cidrs)
        own_idx = 0
        other_idx = 0
        own_len = len(own_nets)
        other_len = len(other_nets)
        while own_idx < own_len and other_idx < other_len:
            own_cur = own_nets[own_idx]
            other_cur = other_nets[other_idx]
            if own_cur == other_cur:
                own_idx += 1
                other_idx += 1
            elif own_cur in other_cur:
                own_idx = _subtract(other_cur, own_nets, own_idx, result_ranges)
                other_idx += 1
            elif other_cur in own_cur:
                other_idx = _subtract(own_cur, other_nets, other_idx, result_ranges)
                own_idx += 1
            elif own_cur < other_cur:
                result_ranges.append((own_cur._module.version, own_cur.first, own_cur.last))
                own_idx += 1
            else:
                result_ranges.append((other_cur._module.version, other_cur.first, other_cur.last))
                other_idx += 1
        while own_idx < own_len:
            own_cur = own_nets[own_idx]
            result_ranges.append((own_cur._module.version, own_cur.first, own_cur.last))
            own_idx += 1
        while other_idx < other_len:
            other_cur = other_nets[other_idx]
            result_ranges.append((other_cur._module.version, other_cur.first, other_cur.last))
            other_idx += 1
        result = IPSet()
        for start, stop in _iter_merged_ranges(result_ranges):
            cidrs = iprange_to_cidrs(start, stop)
            for cidr in cidrs:
                result._cidrs[cidr] = True
        return result
    __xor__ = symmetric_difference

    def difference(self, other):
        """
        :param other: an IP set.

        :return: the difference between this IP set and another as a new IP
            set (all IP addresses and subnets that are in this IP set but
            not found in the other.)
        """
        result_ranges = []
        result_cidrs = {}
        own_nets = sorted(self._cidrs)
        other_nets = sorted(other._cidrs)
        own_idx = 0
        other_idx = 0
        own_len = len(own_nets)
        other_len = len(other_nets)
        while own_idx < own_len and other_idx < other_len:
            own_cur = own_nets[own_idx]
            other_cur = other_nets[other_idx]
            if own_cur == other_cur:
                own_idx += 1
                other_idx += 1
            elif own_cur in other_cur:
                own_idx += 1
            elif other_cur in own_cur:
                other_idx = _subtract(own_cur, other_nets, other_idx, result_ranges)
                own_idx += 1
            elif own_cur < other_cur:
                result_cidrs[own_cur] = True
                own_idx += 1
            else:
                other_idx += 1
        while own_idx < own_len:
            result_cidrs[own_nets[own_idx]] = True
            own_idx += 1
        for start, stop in _iter_merged_ranges(result_ranges):
            for cidr in iprange_to_cidrs(start, stop):
                result_cidrs[cidr] = True
        result = IPSet()
        result._cidrs = result_cidrs
        return result
    __sub__ = difference

    def __len__(self):
        """
        :return: the cardinality of this IP set (i.e. sum of individual IP             addresses). Raises ``IndexError`` if size > maxsize (a Python             limitation). Use the .size property for subnets of any size.
        """
        size = self.size
        if size > _sys.maxsize:
            raise IndexError('range contains more than %d (sys.maxsize) IP addresses!Use the .size property instead.' % _sys.maxsize)
        return size

    @property
    def size(self):
        """
        The cardinality of this IP set (based on the number of individual IP
        addresses including those implicitly defined in subnets).
        """
        return sum([cidr.size for cidr in self._cidrs])

    def __repr__(self):
        """:return: Python statement to create an equivalent object"""
        return 'IPSet(%r)' % [str(c) for c in sorted(self._cidrs)]
    __str__ = __repr__

    def iscontiguous(self):
        """
        Returns True if the members of the set form a contiguous IP
        address range (with no gaps), False otherwise.

        :return: ``True`` if the ``IPSet`` object is contiguous.
        """
        cidrs = self.iter_cidrs()
        if len(cidrs) > 1:
            previous = cidrs[0][0]
            for cidr in cidrs:
                if cidr[0] != previous:
                    return False
                previous = cidr[-1] + 1
        return True

    def iprange(self):
        """
        Generates an IPRange for this IPSet, if all its members
        form a single contiguous sequence.

        Raises ``ValueError`` if the set is not contiguous.

        :return: An ``IPRange`` for all IPs in the IPSet.
        """
        if self.iscontiguous():
            cidrs = self.iter_cidrs()
            if not cidrs:
                return None
            return IPRange(cidrs[0][0], cidrs[-1][-1])
        else:
            raise ValueError('IPSet is not contiguous')

    def iter_ipranges(self):
        """Generate the merged IPRanges for this IPSet.

        In contrast to self.iprange(), this will work even when the IPSet is
        not contiguous. Adjacent IPRanges will be merged together, so you
        get the minimal number of IPRanges.
        """
        sorted_ranges = [(cidr._module.version, cidr.first, cidr.last) for cidr in self.iter_cidrs()]
        for start, stop in _iter_merged_ranges(sorted_ranges):
            yield IPRange(start, stop)