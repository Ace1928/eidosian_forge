import copy
import hashlib
import json
import fixtures
from stevedore import extension
from heat.common import exception
from heat.common import template_format
from heat.engine.cfn import functions as cfn_funcs
from heat.engine.cfn import parameters as cfn_p
from heat.engine.cfn import template as cfn_t
from heat.engine.clients.os import nova
from heat.engine import environment
from heat.engine import function
from heat.engine.hot import template as hot_t
from heat.engine import node_data
from heat.engine import rsrc_defn
from heat.engine import stack
from heat.engine import stk_defn
from heat.engine import template
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_resource_facade(self):
    metadata_snippet = {'Fn::ResourceFacade': 'Metadata'}
    deletion_policy_snippet = {'Fn::ResourceFacade': 'DeletionPolicy'}
    update_policy_snippet = {'Fn::ResourceFacade': 'UpdatePolicy'}
    parent_resource = DummyClass()
    parent_resource.metadata_set({'foo': 'bar'})
    parent_resource.t = rsrc_defn.ResourceDefinition('parent', 'SomeType', deletion_policy=rsrc_defn.ResourceDefinition.RETAIN, update_policy={'blarg': 'wibble'})
    tmpl = copy.deepcopy(empty_template)
    tmpl['Resources'] = {'parent': {'Type': 'SomeType', 'DeletionPolicy': 'Retain', 'UpdatePolicy': {'blarg': 'wibble'}}}
    parent_resource.stack = stack.Stack(self.ctx, 'toplevel_stack', template.Template(tmpl))
    parent_resource.stack._resources = {'parent': parent_resource}
    stk = stack.Stack(self.ctx, 'test_stack', template.Template(empty_template), parent_resource='parent', owner_id=45)
    stk.set_parent_stack(parent_resource.stack)
    self.assertEqual({'foo': 'bar'}, self.resolve(metadata_snippet, stk.t, stk))
    self.assertEqual('Retain', self.resolve(deletion_policy_snippet, stk.t, stk))
    self.assertEqual({'blarg': 'wibble'}, self.resolve(update_policy_snippet, stk.t, stk))