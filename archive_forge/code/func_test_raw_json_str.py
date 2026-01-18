import unittest
import simplejson as json
def test_raw_json_str(self):
    self.assertEqual(json.dumps(dct2), json.dumps(dct4))
    self.assertEqual(dct2, json.loads(json.dumps(dct4)))