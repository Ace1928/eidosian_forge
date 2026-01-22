from tests.unit import unittest
from mock import call, Mock, patch, sentinel
import codecs
from boto.glacier.layer1 import Layer1
from boto.glacier.layer2 import Layer2
import boto.glacier.vault
from boto.glacier.vault import Vault
from boto.glacier.vault import Job
from datetime import datetime, tzinfo, timedelta
class TestRangeStringParsing(unittest.TestCase):

    def test_simple_range(self):
        self.assertEquals(Vault._range_string_to_part_index('0-3', 4), 0)

    def test_range_one_too_big(self):
        self.assertEquals(Vault._range_string_to_part_index('0-4', 4), 0)

    def test_range_too_big(self):
        self.assertRaises(AssertionError, Vault._range_string_to_part_index, '0-5', 4)

    def test_range_start_mismatch(self):
        self.assertRaises(AssertionError, Vault._range_string_to_part_index, '1-3', 4)

    def test_range_end_mismatch(self):
        self.assertEquals(Vault._range_string_to_part_index('0-2', 4), 0)