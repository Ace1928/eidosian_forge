import contextlib
import io
import socket
from unittest import mock
import netaddr
import netifaces
from oslotest import base as test_base
from oslo_utils import netutils
def test_random_qemu_mac(self):
    self.assertEqual(netaddr.EUI('52:54:00:42:02:19'), netutils.get_mac_addr_by_ipv6(netaddr.IPAddress('fe80::5054:ff:fe42:219')))