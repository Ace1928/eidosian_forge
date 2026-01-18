from unittest import TestCase
from simplejson.compat import StringIO, long_type, b, binary_type, text_type, PY3
import simplejson as json
def test_stringify_key(self):
    items = [(b('bytes'), 'bytes'), (1.0, '1.0'), (10, '10'), (True, 'true'), (False, 'false'), (None, 'null'), (long_type(100), '100')]
    for k, expect in items:
        self.assertEqual(json.loads(json.dumps({k: expect})), {expect: expect})
        self.assertEqual(json.loads(json.dumps({k: expect}, sort_keys=True)), {expect: expect})
    self.assertRaises(TypeError, json.dumps, {json: 1})
    for v in [{}, {'other': 1}, {b('derp'): 1, 'herp': 2}]:
        for sort_keys in [False, True]:
            v0 = dict(v)
            v0[json] = 1
            v1 = dict(((as_text_type(key), val) for key, val in v.items()))
            self.assertEqual(json.loads(json.dumps(v0, skipkeys=True, sort_keys=sort_keys)), v1)
            self.assertEqual(json.loads(json.dumps({'': v0}, skipkeys=True, sort_keys=sort_keys)), {'': v1})
            self.assertEqual(json.loads(json.dumps([v0], skipkeys=True, sort_keys=sort_keys)), [v1])