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
def test_update_rollback(self):
    tmpl = {'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'AResource': {'Type': 'ResourceWithPropsType', 'Properties': {'Foo': 'abc'}}}}
    self.stack = stack.Stack(self.ctx, 'update_test_stack', template.Template(tmpl), disable_rollback=False)
    self.stack.store()
    self.stack.create()
    self.assertEqual((stack.Stack.CREATE, stack.Stack.COMPLETE), self.stack.state)
    self.stack._persist_state()
    tmpl2 = {'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'AResource': {'Type': 'ResourceWithPropsType', 'Properties': {'Foo': 'xyz'}}}}
    updated_stack = stack.Stack(self.ctx, 'updated_stack', template.Template(tmpl2), disable_rollback=False)
    mock_create = self.patchobject(generic_rsrc.ResourceWithProps, 'handle_create', side_effect=Exception)
    with mock.patch.object(stack_object.Stack, 'update_by_id', wraps=stack_object.Stack.update_by_id) as mock_db_update:
        self.stack.update(updated_stack)
        self.assertEqual((stack.Stack.ROLLBACK, stack.Stack.COMPLETE), self.stack.state)
        self.eng = service.EngineService('a-host', 'a-topic')
        events = self.eng.list_events(self.ctx, self.stack.identifier())
        self.assertEqual(11, len(events))
        self.assertEqual('abc', self.stack['AResource']._stored_properties_data['Foo'])
        self.assertEqual(5, mock_db_update.call_count)
        self.assertEqual('UPDATE', mock_db_update.call_args_list[0][0][2]['action'])
        self.assertEqual('IN_PROGRESS', mock_db_update.call_args_list[0][0][2]['status'])
        self.assertEqual('ROLLBACK', mock_db_update.call_args_list[1][0][2]['action'])
        self.assertEqual('IN_PROGRESS', mock_db_update.call_args_list[1][0][2]['status'])
    mock_create.assert_called_once_with()