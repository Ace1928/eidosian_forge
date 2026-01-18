import os
from os.path import abspath, dirname
from pyomo.common import DeveloperError
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.environ import ConcreteModel, Var, Param, Set, value, Integers
from pyomo.core.base.set import FiniteSetOf, OrderedSetOf
from pyomo.core.base.indexed_component import normalize_index
from pyomo.core.expr import GetItemExpression
from pyomo.core import SortComponents
def test_ordered_items_deprecation(self):
    m = ConcreteModel()
    unordered = [1, 3, 2]
    ordered = [1, 2, 3]
    m.I = FiniteSetOf(unordered)
    m.x = Var(m.I)
    unordered = [(i, m.x[i]) for i in unordered]
    ordered = [(i, m.x[i]) for i in ordered]
    self.assertEqual(list(m.x.items()), unordered)
    self.assertEqual(list(m.x.items(SortComponents.ORDERED_INDICES)), ordered)
    with LoggingIntercept() as LOG:
        self.assertEqual(list(m.x.items(True)), ordered)
    self.assertEqual(LOG.getvalue(), '')
    with LoggingIntercept() as LOG:
        self.assertEqual(list(m.x.items(ordered=True)), ordered)
    self.assertIn('items(ordered=True) is deprecated', LOG.getvalue())
    with LoggingIntercept() as LOG:
        self.assertEqual(list(m.x.items(ordered=False)), unordered)
    self.assertIn('items(ordered=False) is deprecated', LOG.getvalue())
    m = ConcreteModel()
    unordered = [1, 3, 2]
    ordered = [1, 2, 3]
    m.I = OrderedSetOf(unordered)
    m.x = Var(m.I)
    unordered = [(i, m.x[i]) for i in unordered]
    ordered = [(i, m.x[i]) for i in ordered]
    self.assertEqual(list(m.x.items()), unordered)
    self.assertEqual(list(m.x.items(SortComponents.ORDERED_INDICES)), unordered)
    with LoggingIntercept() as LOG:
        self.assertEqual(list(m.x.items(True)), ordered)
    self.assertEqual(LOG.getvalue(), '')
    with LoggingIntercept() as LOG:
        self.assertEqual(list(m.x.items(ordered=True)), unordered)
    self.assertIn('items(ordered=True) is deprecated', LOG.getvalue())
    with LoggingIntercept() as LOG:
        self.assertEqual(list(m.x.items(ordered=False)), unordered)
    self.assertIn('items(ordered=False) is deprecated', LOG.getvalue())