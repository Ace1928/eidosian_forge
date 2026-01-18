import copy
from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.db import api as db_api
from heat.engine.clients.os.keystone import fake_keystoneclient
from heat.engine import environment
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import service
from heat.engine import stack
from heat.engine import support
from heat.engine import template
from heat.objects import stack as stack_object
from heat.rpc import api as rpc_api
from heat.tests import common
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
def test_update_modify_ok_replace_int(self):
    tmpl = {'heat_template_version': '2013-05-23', 'resources': {'AResource': {'type': 'ResWithComplexPropsAndAttrs', 'properties': {'an_int': 1}}}}
    self.stack = stack.Stack(self.ctx, 'update_test_stack', template.Template(tmpl))
    self.stack.store()
    stack_id = self.stack.id
    self.stack.create()
    self.stack._persist_state()
    self.assertEqual((stack.Stack.CREATE, stack.Stack.COMPLETE), self.stack.state)
    value1 = 2
    prop_diff1 = {'an_int': value1}
    value2 = 1
    prop_diff2 = {'an_int': value2}
    mock_upd = self.patchobject(generic_rsrc.ResWithComplexPropsAndAttrs, 'handle_update')
    self.stack = stack.Stack.load(self.ctx, stack_id=stack_id)
    tmpl2 = {'heat_template_version': '2013-05-23', 'resources': {'AResource': {'type': 'ResWithComplexPropsAndAttrs', 'properties': {'an_int': value1}}}}
    updated_stack = stack.Stack(self.ctx, 'updated_stack', template.Template(tmpl2))
    self.stack.update(updated_stack)
    self.stack._persist_state()
    self.assertEqual((stack.Stack.UPDATE, stack.Stack.COMPLETE), self.stack.state)
    mock_upd.assert_called_once_with(mock.ANY, mock.ANY, prop_diff1)
    self.stack = stack.Stack.load(self.ctx, stack_id=stack_id)
    tmpl3 = {'heat_template_version': '2013-05-23', 'resources': {'AResource': {'type': 'ResWithComplexPropsAndAttrs', 'properties': {'an_int': value2}}}}
    updated_stack = stack.Stack(self.ctx, 'updated_stack', template.Template(tmpl3))
    self.stack.update(updated_stack)
    self.stack._persist_state()
    self.assertEqual((stack.Stack.UPDATE, stack.Stack.COMPLETE), self.stack.state)
    mock_upd.assert_called_with(mock.ANY, mock.ANY, prop_diff2)