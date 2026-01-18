import unittest
import logging
import inspect
import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import cfm
def test_parser_with_cc_message(self):
    self.setUp_cc_message()
    self.test_parser()