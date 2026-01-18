import pickle
import pytest
from netaddr import IPAddress, IPNetwork
def test_rfc6164_subnets():
    assert list(IPNetwork('1234::/127')) == [IPAddress('1234::'), IPAddress('1234::1')]
    assert list(IPNetwork('1234::/127').iter_hosts()) == [IPAddress('1234::'), IPAddress('1234::1')]
    assert IPNetwork('1234::/127').network == IPAddress('1234::')
    assert IPNetwork('1234::').broadcast is None
    assert IPNetwork('1234::/128').network == IPAddress('1234::')
    assert IPNetwork('1234::/128').broadcast is None
    assert list(IPNetwork('1234::/128')) == [IPAddress('1234::')]
    assert list(IPNetwork('1234::/128').iter_hosts()) == [IPAddress('1234::')]