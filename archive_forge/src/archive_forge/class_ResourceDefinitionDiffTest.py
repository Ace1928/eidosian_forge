from heat.common import exception
from heat.common import template_format
from heat.engine.cfn import functions as cfn_funcs
from heat.engine import function
from heat.engine.hot import functions as hot_funcs
from heat.engine import properties
from heat.engine import rsrc_defn
from heat.tests import common
from heat.tests import utils
class ResourceDefinitionDiffTest(common.HeatTestCase):

    def test_properties_diff(self):
        before = rsrc_defn.ResourceDefinition('rsrc', 'SomeType', properties={'Foo': 'blarg'}, update_policy={'baz': 'quux'}, metadata={'baz': 'quux'})
        after = rsrc_defn.ResourceDefinition('rsrc', 'SomeType', properties={'Foo': 'wibble'}, update_policy={'baz': 'quux'}, metadata={'baz': 'quux'})
        diff = after - before
        self.assertTrue(diff.properties_changed())
        self.assertFalse(diff.update_policy_changed())
        self.assertFalse(diff.metadata_changed())
        self.assertTrue(diff)

    def test_update_policy_diff(self):
        before = rsrc_defn.ResourceDefinition('rsrc', 'SomeType', properties={'baz': 'quux'}, update_policy={'Foo': 'blarg'}, metadata={'baz': 'quux'})
        after = rsrc_defn.ResourceDefinition('rsrc', 'SomeType', properties={'baz': 'quux'}, update_policy={'Foo': 'wibble'}, metadata={'baz': 'quux'})
        diff = after - before
        self.assertFalse(diff.properties_changed())
        self.assertTrue(diff.update_policy_changed())
        self.assertFalse(diff.metadata_changed())
        self.assertTrue(diff)

    def test_metadata_diff(self):
        before = rsrc_defn.ResourceDefinition('rsrc', 'SomeType', properties={'baz': 'quux'}, update_policy={'baz': 'quux'}, metadata={'Foo': 'blarg'})
        after = rsrc_defn.ResourceDefinition('rsrc', 'SomeType', properties={'baz': 'quux'}, update_policy={'baz': 'quux'}, metadata={'Foo': 'wibble'})
        diff = after - before
        self.assertFalse(diff.properties_changed())
        self.assertFalse(diff.update_policy_changed())
        self.assertTrue(diff.metadata_changed())
        self.assertTrue(diff)

    def test_no_diff(self):
        before = rsrc_defn.ResourceDefinition('rsrc', 'SomeType', properties={'Foo': 'blarg'}, update_policy={'bar': 'quux'}, metadata={'baz': 'wibble'}, depends=['other_resource'], deletion_policy='Delete')
        after = rsrc_defn.ResourceDefinition('rsrc', 'SomeType', properties={'Foo': 'blarg'}, update_policy={'bar': 'quux'}, metadata={'baz': 'wibble'}, depends=['other_other_resource'], deletion_policy='Retain')
        diff = after - before
        self.assertFalse(diff.properties_changed())
        self.assertFalse(diff.update_policy_changed())
        self.assertFalse(diff.metadata_changed())
        self.assertFalse(diff)