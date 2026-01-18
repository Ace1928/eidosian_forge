import pickle
import types
import random
import pytest
from netaddr import (
def test_spanning_cidr_handles_strings():
    addresses = [IPAddress('10.0.0.1'), IPAddress('10.0.0.2'), '10.0.0.3', '10.0.0.4']
    assert spanning_cidr(addresses) == IPNetwork('10.0.0.0/29')
    assert spanning_cidr(reversed(addresses)) == IPNetwork('10.0.0.0/29')