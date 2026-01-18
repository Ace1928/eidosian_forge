import unittest
import os_ken.ofproto.ofproto_v1_3 as ofp
def test_nxm_1_mask(self):
    user = ('tun_ipv4_src', ('192.0.2.1', '255.255.0.0'))
    on_wire = b'\x00\x01?\x08\xc0\x00\x02\x01\xff\xff\x00\x00'
    self._test(user, on_wire, 4)