import contextlib
import io
import socket
from unittest import mock
import netaddr
import netifaces
from oslotest import base as test_base
from oslo_utils import netutils
def test_universal(self):
    self.assertEqual(netaddr.EUI('00:00:00:00:00:00'), netutils.get_mac_addr_by_ipv6(netaddr.IPAddress('fe80::200:ff:fe00:0')))