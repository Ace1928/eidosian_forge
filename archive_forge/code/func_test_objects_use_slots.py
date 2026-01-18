import pickle
import pytest
from netaddr import IPAddress, IPNetwork
def test_objects_use_slots():
    assert not hasattr(IPNetwork('::/64'), '__dict__')
    assert not hasattr(IPAddress('::'), '__dict__')