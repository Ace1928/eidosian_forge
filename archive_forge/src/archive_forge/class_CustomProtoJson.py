import datetime
import json
import unittest
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.protorpclite import test_util
class CustomProtoJson(protojson.ProtoJson):

    def encode_field(self, field, value):
        return '{encoded}' + value

    def decode_field(self, field, value):
        return '{decoded}' + value