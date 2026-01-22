import collections
import operator
import re
import warnings
import abc
from debtcollector import removals
import netaddr
import rfc3986
def _check_ipv4(self, address):
    if not netaddr.valid_ipv4(address, netaddr.core.INET_PTON):
        raise ValueError('%s is not an IPv4 address' % address)