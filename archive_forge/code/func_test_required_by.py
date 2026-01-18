from heat.common import exception
from heat.engine import dependencies
from heat.tests import common
def test_required_by(self):
    d = dependencies.Dependencies([('last', 'e1'), ('last', 'mid1'), ('last', 'mid2'), ('mid1', 'e2'), ('mid1', 'mid3'), ('mid2', 'mid3'), ('mid3', 'e3')])
    self.assertEqual(0, len(list(d.required_by('last'))))
    required_by = list(d.required_by('mid3'))
    self.assertEqual(2, len(required_by))
    for n in ('mid1', 'mid2'):
        self.assertIn(n, required_by, "'%s' not found in required_by" % n)
    required_by = list(d.required_by('e2'))
    self.assertEqual(1, len(required_by))
    self.assertIn('mid1', required_by, "'%s' not found in required_by" % n)
    self.assertRaises(KeyError, d.required_by, 'foo')