import unittest
import logging
import inspect
import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import cfm
def test_serialize_with_link_trace_message(self):
    self.setUp_link_trace_message()
    self.test_serialize()
    data = bytearray()
    prev = None
    buf = self.ins.serialize(data, prev)
    link_trace_message = cfm.link_trace_message.parser(bytes(buf))
    self.assertEqual(repr(self.message), repr(link_trace_message))