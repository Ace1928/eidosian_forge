import unittest
import logging
from os_ken.ofproto.ofproto_v1_0_parser import *
from os_ken.ofproto.nx_actions import *
from os_ken.ofproto import ofproto_v1_0_parser
from os_ken.lib import addrconv
class TestOFPVendor(unittest.TestCase):
    """ Test case for ofproto_v1_0_parser.OFPVendor
    """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        pass

    def test_parser(self):
        version = {'buf': b'\x01', 'val': ofproto.OFP_VERSION}
        msg_type = {'buf': b'\x04', 'val': ofproto.OFPT_VENDOR}
        msg_len = {'buf': b'\x00\x0c', 'val': ofproto.OFP_VENDOR_HEADER_SIZE}
        xid = {'buf': b'\x05E\xdf\x18', 'val': 88465176}
        vendor = {'buf': b'S\xea%>', 'val': 1407853886}
        data = b'Vendor Message.'
        buf = version['buf'] + msg_type['buf'] + msg_len['buf'] + xid['buf'] + vendor['buf'] + data
        res = OFPVendor.parser(object, version['val'], msg_type['val'], msg_len['val'], xid['val'], buf)
        self.assertEqual(version['val'], res.version)
        self.assertEqual(msg_type['val'], res.msg_type)
        self.assertEqual(msg_len['val'], res.msg_len)
        self.assertEqual(xid['val'], res.xid)
        self.assertEqual(vendor['val'], res.vendor)
        self.assertEqual(data, res.data)

    def test_serialize(self):

        class Datapath(object):
            ofproto = ofproto
            ofproto_parser = ofproto_v1_0_parser
        vendor = {'buf': b'8K\xf9l', 'val': 944503148}
        data = b'Reply Message.'
        c = OFPVendor(Datapath)
        c.vendor = vendor['val']
        c.data = data
        c.serialize()
        self.assertEqual(ofproto.OFP_VERSION, c.version)
        self.assertEqual(ofproto.OFPT_VENDOR, c.msg_type)
        self.assertEqual(0, c.xid)
        self.assertEqual(vendor['val'], c.vendor)
        fmt = '!' + ofproto.OFP_HEADER_PACK_STR.replace('!', '') + ofproto.OFP_VENDOR_HEADER_PACK_STR.replace('!', '') + str(len(data)) + 's'
        res = struct.unpack(fmt, bytes(c.buf))
        self.assertEqual(ofproto.OFP_VERSION, res[0])
        self.assertEqual(ofproto.OFPT_VENDOR, res[1])
        self.assertEqual(len(c.buf), res[2])
        self.assertEqual(0, res[3])
        self.assertEqual(vendor['val'], res[4])
        self.assertEqual(data, res[5])