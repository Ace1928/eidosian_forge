import contextlib
import io
import socket
from unittest import mock
import netaddr
import netifaces
from oslotest import base as test_base
from oslo_utils import netutils
def test_urlsplit_ipv6(self):
    ipv6_url = 'http://[::1]:443/v2.0/'
    result = netutils.urlsplit(ipv6_url)
    self.assertEqual(result.scheme, 'http')
    self.assertEqual(result.netloc, '[::1]:443')
    self.assertEqual(result.path, '/v2.0/')
    self.assertEqual(result.hostname, '::1')
    self.assertEqual(result.port, 443)
    ipv6_url = 'http://user:pass@[::1]/v2.0/'
    result = netutils.urlsplit(ipv6_url)
    self.assertEqual(result.scheme, 'http')
    self.assertEqual(result.netloc, 'user:pass@[::1]')
    self.assertEqual(result.path, '/v2.0/')
    self.assertEqual(result.hostname, '::1')
    self.assertIsNone(result.port)
    ipv6_url = 'https://[2001:db8:85a3::8a2e:370:7334]:1234/v2.0/xy?ab#12'
    result = netutils.urlsplit(ipv6_url)
    self.assertEqual(result.scheme, 'https')
    self.assertEqual(result.netloc, '[2001:db8:85a3::8a2e:370:7334]:1234')
    self.assertEqual(result.path, '/v2.0/xy')
    self.assertEqual(result.hostname, '2001:db8:85a3::8a2e:370:7334')
    self.assertEqual(result.port, 1234)
    self.assertEqual(result.query, 'ab')
    self.assertEqual(result.fragment, '12')