import datetime
import json
import unittest
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.protorpclite import test_util
def CompareEncoded(self, expected_encoded, actual_encoded):
    """JSON encoding will be laundered to remove string differences."""
    self.assertEquals(json.loads(expected_encoded), json.loads(actual_encoded))