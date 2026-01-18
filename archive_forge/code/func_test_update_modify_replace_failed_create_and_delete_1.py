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
def test_update_modify_replace_failed_create_and_delete_1(self):
    tmpl = {'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'AResource': {'Type': 'ResourceWithResourceIDType', 'Properties': {'ID': 'a_res'}}, 'BResource': {'Type': 'ResourceWithResourceIDType', 'Properties': {'ID': 'b_res'}, 'DependsOn': 'AResource'}}}
    self.stack = stack.Stack(self.ctx, 'update_test_stack', template.Template(tmpl), disable_rollback=True)
    self.stack.store()
    self.stack.create()
    self.assertEqual((stack.Stack.CREATE, stack.Stack.COMPLETE), self.stack.state)
    tmpl2 = {'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'AResource': {'Type': 'ResourceWithResourceIDType', 'Properties': {'ID': 'xyz'}}, 'BResource': {'Type': 'ResourceWithResourceIDType', 'Properties': {'ID': 'b_res'}, 'DependsOn': 'AResource'}}}
    updated_stack = stack.Stack(self.ctx, 'updated_stack', template.Template(tmpl2))
    mock_create = self.patchobject(generic_rsrc.ResourceWithResourceID, 'handle_create', side_effect=Exception)
    mock_id = self.patchobject(generic_rsrc.ResourceWithResourceID, 'mock_resource_id', return_value=None)
    self.stack.update(updated_stack)
    self.assertEqual((stack.Stack.UPDATE, stack.Stack.FAILED), self.stack.state)
    self.stack.delete()
    self.assertEqual((stack.Stack.DELETE, stack.Stack.COMPLETE), self.stack.state)
    mock_create.assert_called_once_with()
    mock_id.assert_has_calls([mock.call(None), mock.call('b_res'), mock.call('a_res')])