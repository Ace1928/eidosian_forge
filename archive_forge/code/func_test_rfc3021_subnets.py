import pickle
import types
import random
import pytest
from netaddr import (
def test_rfc3021_subnets():
    assert IPNetwork('192.0.2.0/31').network == IPAddress('192.0.2.0')
    assert IPNetwork('192.0.2.0/31').broadcast is None
    assert list(IPNetwork('192.0.2.0/31').iter_hosts()) == [IPAddress('192.0.2.0'), IPAddress('192.0.2.1')]
    assert IPNetwork('192.0.2.0/32').network == IPAddress('192.0.2.0')
    assert IPNetwork('192.0.2.0/32').broadcast is None
    assert list(IPNetwork('192.0.2.0/32').iter_hosts()) == [IPAddress('192.0.2.0')]