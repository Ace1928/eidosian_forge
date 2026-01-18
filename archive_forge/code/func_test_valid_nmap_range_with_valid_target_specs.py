import pytest
from netaddr import valid_nmap_range, iter_nmap_range, IPAddress, AddrFormatError
def test_valid_nmap_range_with_valid_target_specs():
    assert valid_nmap_range('192.0.2.1')
    assert valid_nmap_range('192.0.2.0-31')
    assert valid_nmap_range('192.0.2-3.1-254')
    assert valid_nmap_range('0-255.0-255.0-255.0-255')
    assert valid_nmap_range('192.168.3-5,7.1')
    assert valid_nmap_range('192.168.3-5,7,10-12,13,14.1')
    assert valid_nmap_range('fe80::1')
    assert valid_nmap_range('::')
    assert valid_nmap_range('192.0.2.0/24')