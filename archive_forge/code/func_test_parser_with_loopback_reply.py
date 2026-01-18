import unittest
import logging
import inspect
import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import cfm
def test_parser_with_loopback_reply(self):
    self.setUp_loopback_reply()
    self.test_parser()