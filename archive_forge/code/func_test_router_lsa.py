import unittest
from os_ken.lib.packet import ospf
def test_router_lsa(self):
    link1 = ospf.RouterLSA.Link(id_='10.0.0.1', data='255.255.255.0', type_=ospf.LSA_LINK_TYPE_STUB, metric=10)
    msg = ospf.RouterLSA(id_='192.168.0.1', adv_router='192.168.0.2', links=[link1])
    binmsg = msg.serialize()
    msg2, cls, rest = ospf.LSA.parser(binmsg)
    self.assertEqual(msg.header.checksum, msg2.header.checksum)
    self.assertEqual(str(msg), str(msg2))
    self.assertEqual(rest, b'')