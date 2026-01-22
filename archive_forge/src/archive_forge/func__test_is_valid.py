import unittest
import logging
import struct
import inspect
from os_ken.ofproto import inet
from os_ken.lib.packet import ipv4
from os_ken.lib.packet import ipv6
from os_ken.lib.packet import packet
from os_ken.lib.packet import packet_utils
from os_ken.lib.packet import vrrp
from os_ken.lib import addrconv
def _test_is_valid(self, type_=None, vrid=None, priority=None, max_adver_int=None):
    if type_ is None:
        type_ = self.type_
    if vrid is None:
        vrid = self.vrid
    if priority is None:
        priority = self.priority
    if max_adver_int is None:
        max_adver_int = self.max_adver_int
    vrrp_ = vrrp.vrrpv3.create(type_, vrid, priority, max_adver_int, [self.ip_address])
    return vrrp_.is_valid()