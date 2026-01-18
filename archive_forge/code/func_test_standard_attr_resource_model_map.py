import gc
from sqlalchemy.ext import declarative
from sqlalchemy import orm
import testtools
from neutron_lib.db import standard_attr
from neutron_lib.tests import _base as base
def test_standard_attr_resource_model_map(self):
    rs_map = standard_attr.get_standard_attr_resource_model_map()
    base = self._make_decl_base()

    class MyModel(standard_attr.HasStandardAttributes, standard_attr.model_base.HasId, base):
        api_collections = ['my_resource', 'my_resource2']
        api_sub_resources = ['my_subresource']
    rs_map = standard_attr.get_standard_attr_resource_model_map()
    self.assertEqual(MyModel, rs_map['my_resource'])
    self.assertEqual(MyModel, rs_map['my_resource2'])
    self.assertEqual(MyModel, rs_map['my_subresource'])
    sub_rs_map = standard_attr.get_standard_attr_resource_model_map(include_resources=False, include_sub_resources=True)
    self.assertNotIn('my_resource', sub_rs_map)
    self.assertNotIn('my_resource2', sub_rs_map)
    self.assertEqual(MyModel, sub_rs_map['my_subresource'])
    nosub_rs_map = standard_attr.get_standard_attr_resource_model_map(include_resources=True, include_sub_resources=False)
    self.assertEqual(MyModel, nosub_rs_map['my_resource'])
    self.assertEqual(MyModel, nosub_rs_map['my_resource2'])
    self.assertNotIn('my_subresource', nosub_rs_map)

    class Dup(standard_attr.HasStandardAttributes, standard_attr.model_base.HasId, base):
        api_collections = ['my_resource']
    with testtools.ExpectedException(RuntimeError):
        standard_attr.get_standard_attr_resource_model_map()