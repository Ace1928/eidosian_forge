import unittest
import logging
import inspect
import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import cfm
def test_serialize_with_cc_message(self):
    self.setUp_cc_message()
    self.test_serialize()
    data = bytearray()
    prev = None
    buf = self.ins.serialize(data, prev)
    cc_message = cfm.cc_message.parser(bytes(buf))
    self.assertEqual(repr(self.message), repr(cc_message))