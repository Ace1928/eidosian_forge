import unittest
from simplejson.compat import StringIO
import simplejson as json
class TestIterable(unittest.TestCase):

    def test_iterable(self):
        for l in ([], [1], [1, 2], [1, 2, 3]):
            for opts in [{}, {'indent': 2}]:
                for dumps in (json.dumps, iter_dumps, sio_dump):
                    expect = dumps(l, **opts)
                    default_expect = dumps(sum(l), **opts)
                    self.assertRaises(TypeError, dumps, iter(l), **opts)
                    self.assertRaises(TypeError, dumps, iter(l), iterable_as_array=False, **opts)
                    self.assertEqual(expect, dumps(iter(l), iterable_as_array=True, **opts))
                    self.assertEqual(default_expect, dumps(iter(l), default=sum, **opts))
                    self.assertEqual(default_expect, dumps(iter(l), iterable_as_array=False, default=sum, **opts))
                    self.assertEqual(expect, dumps(iter(l), iterable_as_array=True, default=sum, **opts))