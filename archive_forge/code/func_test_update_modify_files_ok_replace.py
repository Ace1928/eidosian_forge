import copy
from unittest import mock
from heat.common import exception
from heat.common import identifier
from heat.common import template_format
from heat.engine.cfn import functions as cfn_functions
from heat.engine.cfn import parameters as cfn_param
from heat.engine import conditions
from heat.engine import environment
from heat.engine import function
from heat.engine.hot import functions as hot_functions
from heat.engine.hot import parameters as hot_param
from heat.engine.hot import template as hot_template
from heat.engine import resource
from heat.engine import resources
from heat.engine import rsrc_defn
from heat.engine import stack as parser
from heat.engine import stk_defn
from heat.engine import template
from heat.tests import common
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
def test_update_modify_files_ok_replace(self):
    tmpl = {'heat_template_version': '2013-05-23', 'parameters': {}, 'resources': {'AResource': {'type': 'ResourceWithPropsType', 'properties': {'Foo': {'get_file': 'foo'}}}}}
    self.stack = parser.Stack(self.ctx, 'update_test_stack', template.Template(tmpl, files={'foo': 'abc'}))
    self.stack.store()
    self.stack.create()
    self.assertEqual((parser.Stack.CREATE, parser.Stack.COMPLETE), self.stack.state)
    updated_stack = parser.Stack(self.ctx, 'updated_stack', template.Template(tmpl, files={'foo': 'xyz'}))

    def check_props_and_raise(*args):
        self.assertEqual('abc', self.stack['AResource'].properties['Foo'])
        raise resource.UpdateReplace()
    mock_update = self.patchobject(generic_rsrc.ResourceWithProps, 'update_template_diff', side_effect=check_props_and_raise)
    self.stack.update(updated_stack)
    self.assertEqual((parser.Stack.UPDATE, parser.Stack.COMPLETE), self.stack.state)
    self.assertEqual('xyz', self.stack['AResource'].properties['Foo'])
    mock_update.assert_called_once_with(rsrc_defn.ResourceDefinition('AResource', 'ResourceWithPropsType', properties={'Foo': 'xyz'}), rsrc_defn.ResourceDefinition('AResource', 'ResourceWithPropsType', properties={'Foo': 'abc'}))