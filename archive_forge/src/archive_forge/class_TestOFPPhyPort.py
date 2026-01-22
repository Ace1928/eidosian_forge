import unittest
import logging
from os_ken.ofproto.ofproto_v1_0_parser import *
from os_ken.ofproto.nx_actions import *
from os_ken.ofproto import ofproto_v1_0_parser
from os_ken.lib import addrconv
class TestOFPPhyPort(unittest.TestCase):
    """ Test case for ofproto_v1_0_parser.OFPPhyPort
    """
    port_no = {'buf': b'\xe7k', 'val': 59243}
    hw_addr = '52:54:54:10:20:99'
    name = b'name'.ljust(16)
    config = {'buf': b'\x84\xb6\x8cS', 'val': 2226555987}
    state = {'buf': b'd\x07\xfb\xc9', 'val': 1678244809}
    curr = {'buf': b'\xa9\xe8\n+', 'val': 2850556459}
    advertised = {'buf': b'x\xb9{r', 'val': 2025421682}
    supported = {'buf': b'~eh\xad', 'val': 2120575149}
    peer = {'buf': b'\xa4[\x8b\xed', 'val': 2757463021}
    buf = port_no['buf'] + addrconv.mac.text_to_bin(hw_addr) + name + config['buf'] + state['buf'] + curr['buf'] + advertised['buf'] + supported['buf'] + peer['buf']
    c = OFPPhyPort(port_no['val'], hw_addr, name, config['val'], state['val'], curr['val'], advertised['val'], supported['val'], peer['val'])

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        self.assertEqual(self.port_no['val'], self.c.port_no)
        self.assertEqual(self.hw_addr, self.c.hw_addr)
        self.assertEqual(self.name, self.c.name)
        self.assertEqual(self.config['val'], self.c.config)
        self.assertEqual(self.state['val'], self.c.state)
        self.assertEqual(self.curr['val'], self.c.curr)
        self.assertEqual(self.advertised['val'], self.c.advertised)
        self.assertEqual(self.supported['val'], self.c.supported)
        self.assertEqual(self.peer['val'], self.c.peer)

    def test_parser(self):
        res = self.c.parser(self.buf, 0)
        self.assertEqual(self.port_no['val'], res.port_no)
        self.assertEqual(self.hw_addr, res.hw_addr)
        self.assertEqual(self.name, res.name)
        self.assertEqual(self.config['val'], res.config)
        self.assertEqual(self.state['val'], res.state)
        self.assertEqual(self.curr['val'], res.curr)
        self.assertEqual(self.advertised['val'], res.advertised)
        self.assertEqual(self.supported['val'], res.supported)
        self.assertEqual(self.peer['val'], res.peer)