import contextlib
import io
import socket
from unittest import mock
import netaddr
import netifaces
from oslotest import base as test_base
from oslo_utils import netutils
def test_parse_host_port(self):
    self.assertEqual(('server01', 80), netutils.parse_host_port('server01:80'))
    self.assertEqual(('server01', None), netutils.parse_host_port('server01'))
    self.assertEqual(('server01', 1234), netutils.parse_host_port('server01', default_port=1234))
    self.assertEqual(('::1', 80), netutils.parse_host_port('[::1]:80'))
    self.assertEqual(('::1', None), netutils.parse_host_port('[::1]'))
    self.assertEqual(('::1', 1234), netutils.parse_host_port('[::1]', default_port=1234))
    self.assertEqual(('2001:db8:85a3::8a2e:370:7334', 1234), netutils.parse_host_port('2001:db8:85a3::8a2e:370:7334', default_port=1234))