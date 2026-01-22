import collections
import operator
import re
import warnings
import abc
from debtcollector import removals
import netaddr
import rfc3986
def _check_ipv6(self, address):
    if not netaddr.valid_ipv6(address, netaddr.core.INET_PTON):
        raise ValueError('%s is not an IPv6 address' % address)