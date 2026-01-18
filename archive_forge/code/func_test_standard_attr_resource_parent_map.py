import gc
from sqlalchemy.ext import declarative
from sqlalchemy import orm
import testtools
from neutron_lib.db import standard_attr
from neutron_lib.tests import _base as base
def test_standard_attr_resource_parent_map(self):
    base = self._make_decl_base()

    class TagSupportModel(standard_attr.HasStandardAttributes, standard_attr.model_base.HasId, base):
        collection_resource_map = {'collection_name': 'member_name'}
        tag_support = True

    class TagUnsupportModel(standard_attr.HasStandardAttributes, standard_attr.model_base.HasId, base):
        collection_resource_map = {'collection_name2': 'member_name2'}
        tag_support = False

    class TagUnsupportModel2(standard_attr.HasStandardAttributes, standard_attr.model_base.HasId, base):
        collection_resource_map = {'collection_name3': 'member_name3'}
    parent_map = standard_attr.get_tag_resource_parent_map()
    self.assertEqual('member_name', parent_map['collection_name'])
    self.assertNotIn('collection_name2', parent_map)
    self.assertNotIn('collection_name3', parent_map)

    class DupTagSupportModel(standard_attr.HasStandardAttributes, standard_attr.model_base.HasId, base):
        collection_resource_map = {'collection_name': 'member_name'}
        tag_support = True
    with testtools.ExpectedException(RuntimeError):
        standard_attr.get_tag_resource_parent_map()