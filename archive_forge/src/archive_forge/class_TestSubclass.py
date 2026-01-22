from unittest import TestCase
import simplejson as json
from decimal import Decimal
class TestSubclass(TestCase):

    def test_int(self):
        self.assertEqual(json.dumps(AlternateInt(1)), '1')
        self.assertEqual(json.dumps(AlternateInt(-1)), '-1')
        self.assertEqual(json.loads(json.dumps({AlternateInt(1): 1})), {'1': 1})

    def test_float(self):
        self.assertEqual(json.dumps(AlternateFloat(1.0)), '1.0')
        self.assertEqual(json.dumps(AlternateFloat(-1.0)), '-1.0')
        self.assertEqual(json.loads(json.dumps({AlternateFloat(1.0): 1})), {'1.0': 1})