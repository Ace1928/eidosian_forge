import types
from netaddr import IPNetwork, cidr_merge
def test_supernetting():
    ip = IPNetwork('192.0.2.114')
    supernets = ip.supernet(22)
    assert supernets == [IPNetwork('192.0.0.0/22'), IPNetwork('192.0.2.0/23'), IPNetwork('192.0.2.0/24'), IPNetwork('192.0.2.0/25'), IPNetwork('192.0.2.64/26'), IPNetwork('192.0.2.96/27'), IPNetwork('192.0.2.112/28'), IPNetwork('192.0.2.112/29'), IPNetwork('192.0.2.112/30'), IPNetwork('192.0.2.114/31')]