from heat.common import exception
from heat.engine import dependencies
from heat.tests import common
def test_simple_multilevel_partial(self):
    d = dependencies.Dependencies([('last', 'middle'), ('middle', 'target'), ('target', 'first')])
    p = d['target']
    order = list(iter(p))
    self.assertEqual(3, len(order))
    for n in ('last', 'middle', 'target'):
        self.assertIn(n, order, "'%s' not found in dependency order" % n)