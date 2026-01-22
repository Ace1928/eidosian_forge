import pickle
import pyomo.common.unittest as unittest
from pyomo.core.base.range import (
from pyomo.core.base.set import Any
class TestNonNumericRange(unittest.TestCase):

    def test_str(self):
        a = NNR('a')
        aa = NNR('a')
        b = NNR(None)
        self.assertEqual(str(a), '{a}')
        self.assertEqual(str(aa), '{a}')
        self.assertEqual(str(b), '{None}')

    def test_range_relational(self):
        a = NNR('a')
        aa = NNR('a')
        b = NNR(None)
        self.assertTrue(a.issubset(aa))
        self.assertFalse(a.issubset(b))
        self.assertEqual(a, a)
        self.assertEqual(a, aa)
        self.assertNotEqual(a, b)
        c = NR(None, None, 0)
        self.assertFalse(a.issubset(c))
        self.assertFalse(c.issubset(b))
        self.assertNotEqual(a, c)
        self.assertNotEqual(c, a)

    def test_contains(self):
        a = NNR('a')
        b = NNR(None)
        self.assertIn('a', a)
        self.assertNotIn(0, a)
        self.assertNotIn(None, a)
        self.assertNotIn('a', b)
        self.assertNotIn(0, b)
        self.assertIn(None, b)

    def test_range_difference(self):
        a = NNR('a')
        b = NNR(None)
        self.assertEqual(a.range_difference([NNR('a')]), [])
        self.assertEqual(a.range_difference([b]), [NNR('a')])

    def test_range_intersection(self):
        a = NNR('a')
        b = NNR(None)
        self.assertEqual(a.range_intersection([b]), [])
        self.assertEqual(a.range_intersection([NNR('a')]), [NNR('a')])

    def test_info_methods(self):
        a = NNR('a')
        self.assertTrue(a.isdiscrete())
        self.assertTrue(a.isfinite())

    def test_pickle(self):
        a = NNR('a')
        b = pickle.loads(pickle.dumps(a))
        self.assertIsNot(a, b)
        self.assertEqual(a, b)