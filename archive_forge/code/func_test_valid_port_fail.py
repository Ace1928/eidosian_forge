import contextlib
import io
import socket
from unittest import mock
import netaddr
import netifaces
from oslotest import base as test_base
from oslo_utils import netutils
def test_valid_port_fail(self):
    invalid_inputs = ['-32768', '65536', 528491, '528491', '528.491', 'thirty-seven', None]
    for input_str in invalid_inputs:
        self.assertFalse(netutils.is_valid_port(input_str))