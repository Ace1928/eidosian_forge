import contextlib
import io
import socket
from unittest import mock
import netaddr
import netifaces
from oslotest import base as test_base
from oslo_utils import netutils
def test_reverse_generate_IPv6_by_EUI64(self):
    self.assertEqual(netaddr.EUI('00:16:3e:33:44:55'), netutils.get_mac_addr_by_ipv6(netaddr.IPAddress('2001:db8::216:3eff:fe33:4455')))