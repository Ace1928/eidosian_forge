import copy
from unittest import mock
from heat.common import exception
from heat.common import grouputils
from heat.common import template_format
from heat.engine.clients.os import glance
from heat.engine.clients.os import nova
from heat.engine import node_data
from heat.engine.resources.openstack.heat import resource_group
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
class ResourceGroupAttrTest(common.HeatTestCase):

    def test_aggregate_attribs(self):
        """Test attribute aggregation.

        Test attribute aggregation and that we mimic the nested resource's
        attributes.
        """
        resg = self._create_dummy_stack()
        expected = ['0', '1']
        self.assertEqual(expected, resg.FnGetAtt('foo'))
        self.assertEqual(expected, resg.FnGetAtt('Foo'))

    def test_index_dotted_attribs(self):
        """Test attribute aggregation.

        Test attribute aggregation and that we mimic the nested resource's
        attributes.
        """
        resg = self._create_dummy_stack()
        self.assertEqual('0', resg.FnGetAtt('resource.0.Foo'))
        self.assertEqual('1', resg.FnGetAtt('resource.1.Foo'))

    def test_index_path_attribs(self):
        """Test attribute aggregation.

        Test attribute aggregation and that we mimic the nested resource's
        attributes.
        """
        resg = self._create_dummy_stack()
        self.assertEqual('0', resg.FnGetAtt('resource.0', 'Foo'))
        self.assertEqual('1', resg.FnGetAtt('resource.1', 'Foo'))

    def test_index_deep_path_attribs(self):
        """Test attribute aggregation.

        Test attribute aggregation and that we mimic the nested resource's
        attributes.
        """
        resg = self._create_dummy_stack(template_attr, expect_attrs={'0': 2, '1': 2})
        self.assertEqual(2, resg.FnGetAtt('resource.0', 'nested_dict', 'dict', 'b'))
        self.assertEqual(2, resg.FnGetAtt('resource.1', 'nested_dict', 'dict', 'b'))

    def test_aggregate_deep_path_attribs(self):
        """Test attribute aggregation.

        Test attribute aggregation and that we mimic the nested resource's
        attributes.
        """
        resg = self._create_dummy_stack(template_attr, expect_attrs={'0': 3, '1': 3})
        expected = [3, 3]
        self.assertEqual(expected, resg.FnGetAtt('nested_dict', 'list', 2))

    def test_aggregate_refs(self):
        """Test resource id aggregation."""
        resg = self._create_dummy_stack()
        expected = ['ID-0', 'ID-1']
        self.assertEqual(expected, resg.FnGetAtt('refs'))

    def test_aggregate_refs_with_index(self):
        """Test resource id aggregation with index."""
        resg = self._create_dummy_stack()
        expected = ['ID-0', 'ID-1']
        self.assertEqual(expected[0], resg.FnGetAtt('refs', 0))
        self.assertEqual(expected[1], resg.FnGetAtt('refs', 1))
        self.assertIsNone(resg.FnGetAtt('refs', 2))

    def test_aggregate_refs_map(self):
        resg = self._create_dummy_stack()
        found = resg.FnGetAtt('refs_map')
        expected = {'0': 'ID-0', '1': 'ID-1'}
        self.assertEqual(expected, found)

    def test_aggregate_outputs(self):
        """Test outputs aggregation."""
        expected = {'0': ['foo', 'bar'], '1': ['foo', 'bar']}
        resg = self._create_dummy_stack(template_attr, expect_attrs=expected)
        self.assertEqual(expected, resg.FnGetAtt('attributes', 'list'))

    def test_aggregate_outputs_no_path(self):
        """Test outputs aggregation with missing path."""
        resg = self._create_dummy_stack(template_attr)
        self.assertRaises(exception.InvalidTemplateAttribute, resg.FnGetAtt, 'attributes')

    def test_index_refs(self):
        """Tests getting ids of individual resources."""
        resg = self._create_dummy_stack()
        self.assertEqual('ID-0', resg.FnGetAtt('resource.0'))
        self.assertEqual('ID-1', resg.FnGetAtt('resource.1'))
        ex = self.assertRaises(exception.NotFound, resg.FnGetAtt, 'resource.2')
        self.assertIn("Member '2' not found in group resource 'group1'.", str(ex))

    def test_get_attribute_convg(self):
        cache_data = {'group1': node_data.NodeData.from_dict({'uuid': mock.ANY, 'id': mock.ANY, 'action': 'CREATE', 'status': 'COMPLETE', 'attrs': {'refs': ['rsrc1', 'rsrc2']}})}
        stack = utils.parse_stack(template, cache_data=cache_data)
        rsrc = stack.defn['group1']
        self.assertEqual(['rsrc1', 'rsrc2'], rsrc.FnGetAtt('refs'))

    def test_get_attribute_skiplist(self):
        resg = self._create_dummy_stack()
        resg.data = mock.Mock(return_value={'name_blacklist': '3,5'})
        expected = ['3', '5']
        self.assertEqual(expected, resg.FnGetAtt(resg.REMOVED_RSRC_LIST))

    def _create_dummy_stack(self, template_data=template, expect_count=2, expect_attrs=None):
        stack = utils.parse_stack(template_data)
        resg = stack['group1']
        resg.resource_id = 'test-test'
        attrs = {}
        refids = {}
        if expect_attrs is None:
            expect_attrs = {}
        for index in range(expect_count):
            res = str(index)
            attrs[index] = expect_attrs.get(res, res)
            refids[index] = 'ID-%s' % res
        names = [str(name) for name in range(expect_count)]
        resg._resource_names = mock.Mock(return_value=names)
        self._stub_get_attr(resg, refids, attrs)
        return resg

    def _stub_get_attr(self, resg, refids, attrs):

        def ref_id_fn(res_name):
            return refids[int(res_name)]

        def attr_fn(args):
            res_name = args[0]
            return attrs[int(res_name)]

        def get_output(output_name):
            outputs = resg._nested_output_defns(resg._resource_names(), attr_fn, ref_id_fn)
            op_defns = {od.name: od for od in outputs}
            self.assertIn(output_name, op_defns)
            return op_defns[output_name].get_value()
        orig_get_attr = resg.FnGetAtt

        def get_attr(attr_name, *path):
            if not path:
                attr = attr_name
            else:
                attr = (attr_name,) + path
            resg.referenced_attrs = mock.Mock(return_value=[attr])
            return orig_get_attr(attr_name, *path)
        resg.FnGetAtt = mock.Mock(side_effect=get_attr)
        resg.get_output = mock.Mock(side_effect=get_output)