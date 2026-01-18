import pickle
import sys
import weakref
import pytest
from netaddr import IPAddress, IPNetwork, IPRange, IPSet, cidr_exclude
def test_set_membership_smallest():
    ipset = IPSet(['10.0.0.42/32'])
    assert IPAddress('10.0.0.42') in ipset
    assert IPNetwork('10.0.0.42/32') in ipset
    assert IPAddress('10.0.0.41') not in ipset
    assert IPAddress('10.0.0.43') not in ipset
    assert IPNetwork('10.0.0.42/31') not in ipset