import copy
import itertools
import os
from os.path import abspath, dirname
from io import StringIO
import pyomo.common.unittest as unittest
import pyomo.core.base
from pyomo.core.base.util import flatten_tuple
from pyomo.environ import (
from pyomo.core.base.set import _AnySet, RangeDifferenceError
class TestNestedSetOperations(unittest.TestCase):

    def test_union(self):
        model = AbstractModel()
        s1 = set([1, 2])
        model.s1 = Set(initialize=s1)
        s2 = set(['a', 'b'])
        model.s2 = Set(initialize=s2)
        s3 = set([None, True])
        model.s3 = Set(initialize=s3)
        model.union1 = model.s1 | (model.s2 | (model.s3 | (model.s3 | model.s2)))
        model.union2 = model.s1 | (model.s2 | (model.s3 | (model.s3 | model.s2)))
        model.union3 = model.s1 | model.s2 | model.s3 | model.s3 | model.s2
        inst = model.create_instance()
        union = s1 | s2 | s3 | s3 | s2
        self.assertTrue(isinstance(inst.union1, pyomo.core.base.set.SetUnion))
        self.assertEqual(inst.union1, s1 | (s2 | (s3 | (s3 | s2))))
        self.assertTrue(isinstance(inst.union2, pyomo.core.base.set.SetUnion))
        self.assertEqual(inst.union2, s1 | (s2 | (s3 | (s3 | s2))))
        self.assertTrue(isinstance(inst.union3, pyomo.core.base.set.SetUnion))
        self.assertEqual(inst.union3, s1 | s2 | s3 | s3 | s2)

    def test_intersection(self):
        model = AbstractModel()
        s1 = set([1, 2])
        model.s1 = Set(initialize=s1)
        s2 = set(['a', 'b'])
        model.s2 = Set(initialize=s2)
        s3 = set([None, True])
        model.s3 = Set(initialize=s3)
        model.intersection1 = model.s1 & (model.s2 & (model.s3 & (model.s3 & model.s2)))
        model.intersection2 = model.s1 & (model.s2 & (model.s3 & (model.s3 & model.s2)))
        model.intersection3 = model.s1 & model.s2 & model.s3 & model.s3 & model.s2
        model.intersection4 = model.s3 & model.s1 & model.s3
        inst = model.create_instance()
        self.assertTrue(isinstance(inst.intersection1, pyomo.core.base.set.SetIntersection))
        self.assertEqual(sorted(inst.intersection1), sorted(s1 & (s2 & (s3 & (s3 & s2)))))
        self.assertTrue(isinstance(inst.intersection2, pyomo.core.base.set.SetIntersection))
        self.assertEqual(sorted(inst.intersection2), sorted(s1 & (s2 & (s3 & (s3 & s2)))))
        self.assertTrue(isinstance(inst.intersection3, pyomo.core.base.set.SetIntersection))
        self.assertEqual(sorted(inst.intersection3), sorted(s1 & s2 & s3 & s3 & s2))
        self.assertTrue(isinstance(inst.intersection4, pyomo.core.base.set.SetIntersection))
        self.assertEqual(sorted(inst.intersection4), sorted(s3 & s1 & s3))

    def test_difference(self):
        model = AbstractModel()
        s1 = set([1, 2])
        model.s1 = Set(initialize=s1)
        s2 = set(['a', 'b'])
        model.s2 = Set(initialize=s2)
        s3 = set([None, True])
        model.s3 = Set(initialize=s3)
        model.difference1 = model.s1 - (model.s2 - (model.s3 - (model.s3 - model.s2)))
        model.difference2 = model.s1 - (model.s2 - (model.s3 - (model.s3 - model.s2)))
        model.difference3 = model.s1 - model.s2 - model.s3 - model.s3 - model.s2
        inst = model.create_instance()
        self.assertTrue(isinstance(inst.difference1, pyomo.core.base.set.SetDifference))
        self.assertEqual(sorted(inst.difference1), sorted(s1 - (s2 - (s3 - (s3 - s2)))))
        self.assertTrue(isinstance(inst.difference2, pyomo.core.base.set.SetDifference))
        self.assertEqual(sorted(inst.difference2), sorted(s1 - (s2 - (s3 - (s3 - s2)))))
        self.assertTrue(isinstance(inst.difference3, pyomo.core.base.set.SetDifference))
        self.assertEqual(sorted(inst.difference3), sorted(s1 - s2 - s3 - s3 - s2))

    def test_symmetric_difference(self):
        model = AbstractModel()
        s1 = set([1, 2])
        model.s1 = Set(initialize=s1)
        s2 = set([4, 5])
        model.s2 = Set(initialize=s2)
        s3 = set([0, True])
        model.s3 = Set(initialize=s3)
        model.symdiff1 = model.s1 ^ (model.s2 ^ (model.s3 ^ (model.s3 ^ model.s2)))
        model.symdiff2 = model.s1 ^ (model.s2 ^ (model.s3 ^ (model.s3 ^ model.s2)))
        model.symdiff3 = model.s1 ^ model.s2 ^ model.s3 ^ model.s3 ^ model.s2
        model.symdiff4 = model.s1 ^ model.s2 ^ model.s3
        inst = model.create_instance()
        self.assertTrue(isinstance(inst.symdiff1, pyomo.core.base.set.SetSymmetricDifference))
        self.assertEqual(sorted(inst.symdiff1), sorted(s1 ^ (s2 ^ (s3 ^ (s3 ^ s2)))))
        self.assertTrue(isinstance(inst.symdiff2, pyomo.core.base.set.SetSymmetricDifference))
        self.assertEqual(sorted(inst.symdiff2), sorted(s1 ^ (s2 ^ (s3 ^ (s3 ^ s2)))))
        self.assertTrue(isinstance(inst.symdiff3, pyomo.core.base.set.SetSymmetricDifference))
        self.assertEqual(sorted(inst.symdiff3), sorted(s1 ^ s2 ^ s3 ^ s3 ^ s2))
        self.assertTrue(isinstance(inst.symdiff4, pyomo.core.base.set.SetSymmetricDifference))
        self.assertEqual(sorted(inst.symdiff4), sorted(s1 ^ s2 ^ s3))

    def test_product(self):
        model = AbstractModel()
        s1 = set([1, 2])
        model.s1 = Set(initialize=s1)
        s2 = set([4, 5])
        model.s2 = Set(initialize=s2)
        s3 = set([0, True])
        model.s3 = Set(initialize=s3)
        model.product1 = model.s1 * (model.s2 * (model.s3 * (model.s3 * model.s2)))
        model.product2 = model.s1 * (model.s2 * (model.s3 * (model.s3 * model.s2)))
        model.product3 = model.s1 * model.s2 * model.s3 * model.s3 * model.s2
        inst = model.create_instance()
        p = itertools.product
        self.assertTrue(isinstance(inst.product1, pyomo.core.base.set.SetProduct))
        prod1 = set([flatten_tuple(i) for i in set(p(s1, p(s2, p(s3, p(s3, s2)))))])
        self.assertEqual(sorted(inst.product1), sorted(prod1))
        self.assertTrue(isinstance(inst.product2, pyomo.core.base.set.SetProduct))
        prod2 = set([flatten_tuple(i) for i in set(p(s1, p(s2, p(s3, p(s3, s2)))))])
        self.assertEqual(sorted(inst.product2), sorted(prod2))
        self.assertTrue(isinstance(inst.product3, pyomo.core.base.set.SetProduct))
        prod3 = set([flatten_tuple(i) for i in set(p(p(p(p(s1, s2), s3), s3), s2))])
        self.assertEqual(sorted(inst.product3), sorted(prod3))