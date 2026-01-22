import datetime
import json
import unittest
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.protorpclite import test_util
class CustomProtoJsonTest(test_util.TestCase):
    """Tests for serialization overriding functionality."""

    def setUp(self):
        self.protojson = CustomProtoJson()

    def testEncode(self):
        self.assertEqual('{"a_string": "{encoded}xyz"}', self.protojson.encode_message(MyMessage(a_string='xyz')))

    def testDecode(self):
        self.assertEqual(MyMessage(a_string='{decoded}xyz'), self.protojson.decode_message(MyMessage, '{"a_string": "xyz"}'))

    def testDecodeEmptyMessage(self):
        self.assertEqual(MyMessage(a_string='{decoded}'), self.protojson.decode_message(MyMessage, '{"a_string": ""}'))

    def testDefault(self):
        self.assertTrue(protojson.ProtoJson.get_default(), protojson.ProtoJson.get_default())
        instance = CustomProtoJson()
        protojson.ProtoJson.set_default(instance)
        self.assertTrue(instance is protojson.ProtoJson.get_default())