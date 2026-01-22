from heat.engine import node_data
from heat.tests import common
class NodeDataTest(common.HeatTestCase):

    def test_round_trip(self):
        in_dict = make_test_data()
        self.assertEqual(in_dict, node_data.NodeData.from_dict(in_dict).as_dict())

    def test_resource_key(self):
        nd = make_test_node()
        self.assertEqual(42, nd.primary_key)

    def test_resource_name(self):
        nd = make_test_node()
        self.assertEqual('foo', nd.name)

    def test_action(self):
        nd = make_test_node()
        self.assertEqual('CREATE', nd.action)

    def test_status(self):
        nd = make_test_node()
        self.assertEqual('COMPLETE', nd.status)

    def test_refid(self):
        nd = make_test_node()
        self.assertEqual('foo-000000', nd.reference_id())

    def test_all_attrs(self):
        nd = make_test_node()
        self.assertEqual({'foo': 'bar'}, nd.attributes())

    def test_attr(self):
        nd = make_test_node()
        self.assertEqual('bar', nd.attribute('foo'))

    def test_path_attr(self):
        nd = make_test_node()
        self.assertEqual('quux', nd.attribute(('foo', 'bar', 'baz')))

    def test_attr_names(self):
        nd = make_test_node()
        self.assertEqual({'foo', 'blarg'}, set(nd.attribute_names()))