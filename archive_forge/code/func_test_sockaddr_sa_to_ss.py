import unittest
import platform
import sys
from os_ken.lib import sockaddr
def test_sockaddr_sa_to_ss(self):
    addr = b'\x01'
    expected_result = b'\x01' + 127 * b'\x00'
    self.assertEqual(expected_result, sockaddr.sa_to_ss(addr))