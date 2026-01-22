import builtins
import json
import os
import subprocess
import sys
import tempfile
from unittest import mock
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.tests.unit import utils as test_utils
class GetJsonDataTest(test_utils.BaseTestCase):

    def test_success(self):
        result = utils.get_json_data(b'\n{"answer": 42}')
        self.assertEqual({'answer': 42}, result)

    def test_definitely_not_json(self):
        self.assertIsNone(utils.get_json_data(b'0x010x020x03'))

    def test_could_be_json(self):
        self.assertIsNone(utils.get_json_data(b'{"hahaha, just kidding\x00'))