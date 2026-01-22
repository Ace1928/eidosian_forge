import sys as _sys
from netaddr.core import (
from netaddr.strategy import ipv4 as _ipv4, ipv6 as _ipv6
class BaseIP(object):
    """
    An abstract base class for common operations shared between various IP
    related subclasses.

    """
    __slots__ = ('_value', '_module', '__weakref__')

    def __init__(self):
        """Constructor."""
        self._value = None
        self._module = None

    def _set_value(self, value):
        if not isinstance(value, int):
            raise TypeError('int argument expected, not %s' % type(value))
        if not 0 <= value <= self._module.max_int:
            raise AddrFormatError('value out of bounds for an %s address!' % self._module.family_name)
        self._value = value
    value = property(lambda self: self._value, _set_value, doc='a positive integer representing the value of IP address/subnet.')

    def key(self):
        """
        :return: a key tuple that uniquely identifies this IP address.
        """
        return NotImplemented

    def sort_key(self):
        """
        :return: A key tuple used to compare and sort this `IPAddress`
            correctly.
        """
        return NotImplemented

    def __hash__(self):
        """
        :return: A hash value uniquely identifying this IP object.
        """
        return hash(self.key())

    def __eq__(self, other):
        """
        :param other: an `IPAddress` or `IPNetwork` object.

        :return: ``True`` if this `IPAddress` or `IPNetwork` object is
            equivalent to ``other``, ``False`` otherwise.
        """
        try:
            return self.key() == other.key()
        except (AttributeError, TypeError):
            return NotImplemented

    def __ne__(self, other):
        """
        :param other: an `IPAddress` or `IPNetwork` object.

        :return: ``True`` if this `IPAddress` or `IPNetwork` object is
            not equivalent to ``other``, ``False`` otherwise.
        """
        try:
            return self.key() != other.key()
        except (AttributeError, TypeError):
            return NotImplemented

    def __lt__(self, other):
        """
        :param other: an `IPAddress` or `IPNetwork` object.

        :return: ``True`` if this `IPAddress` or `IPNetwork` object is
            less than ``other``, ``False`` otherwise.
        """
        try:
            return self.sort_key() < other.sort_key()
        except (AttributeError, TypeError):
            return NotImplemented

    def __le__(self, other):
        """
        :param other: an `IPAddress` or `IPNetwork` object.

        :return: ``True`` if this `IPAddress` or `IPNetwork` object is
            less than or equal to ``other``, ``False`` otherwise.
        """
        try:
            return self.sort_key() <= other.sort_key()
        except (AttributeError, TypeError):
            return NotImplemented

    def __gt__(self, other):
        """
        :param other: an `IPAddress` or `IPNetwork` object.

        :return: ``True`` if this `IPAddress` or `IPNetwork` object is
            greater than ``other``, ``False`` otherwise.
        """
        try:
            return self.sort_key() > other.sort_key()
        except (AttributeError, TypeError):
            return NotImplemented

    def __ge__(self, other):
        """
        :param other: an `IPAddress` or `IPNetwork` object.

        :return: ``True`` if this `IPAddress` or `IPNetwork` object is
            greater than or equal to ``other``, ``False`` otherwise.
        """
        try:
            return self.sort_key() >= other.sort_key()
        except (AttributeError, TypeError):
            return NotImplemented

    def is_unicast(self):
        """:return: ``True`` if this IP is unicast, ``False`` otherwise"""
        return not self.is_multicast()

    def is_multicast(self):
        """:return: ``True`` if this IP is multicast, ``False`` otherwise"""
        if self._module == _ipv4:
            return self in IPV4_MULTICAST
        elif self._module == _ipv6:
            return self in IPV6_MULTICAST

    def is_loopback(self):
        """
        :return: ``True`` if this IP is loopback address (not for network
            transmission), ``False`` otherwise.
            References: RFC 3330 and 4291.

        .. note:: |ipv4_in_ipv6_handling|
        """
        if self._module.version == 4:
            return self in IPV4_LOOPBACK
        elif self._module.version == 6:
            return self in IPV6_LOOPBACK

    def is_link_local(self):
        """
        :return: ``True`` if this IP is link-local address ``False`` otherwise.
            Reference: RFCs 3927 and 4291.

        .. note:: |ipv4_in_ipv6_handling|
        """
        if self._module.version == 4:
            return self in IPV4_LINK_LOCAL
        elif self._module.version == 6:
            return self in IPV6_LINK_LOCAL

    def is_reserved(self):
        """
        :return: ``True`` if this IP is in IANA reserved range, ``False``
            otherwise. Reference: RFCs 3330 and 3171.

        .. note:: |ipv4_in_ipv6_handling|
        """
        if self._module.version == 4:
            for cidr in IPV4_RESERVED:
                if self in cidr:
                    return True
        elif self._module.version == 6:
            for cidr in IPV6_RESERVED:
                if self in cidr:
                    return True
        return False

    def is_ipv4_mapped(self):
        """
        :return: ``True`` if this IP is IPv4-compatible IPv6 address, ``False``
            otherwise.
        """
        return self._module.version == 6 and self._value >> 32 == 65535

    def is_ipv4_compat(self):
        """
        :return: ``True`` if this IP is IPv4-mapped IPv6 address, ``False``
            otherwise.
        """
        return self._module.version == 6 and self._value >> 32 == 0

    @property
    def info(self):
        """
        A record dict containing IANA registration details for this IP address
        if available, None otherwise.
        """
        from netaddr.ip.iana import query
        return DictDotLookup(query(self))

    @property
    def version(self):
        """the IP protocol version represented by this IP object."""
        return self._module.version