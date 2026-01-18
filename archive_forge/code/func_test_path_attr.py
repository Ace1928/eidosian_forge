from heat.engine import node_data
from heat.tests import common
def test_path_attr(self):
    nd = make_test_node()
    self.assertEqual('quux', nd.attribute(('foo', 'bar', 'baz')))