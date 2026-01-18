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
def test_update_rollback_replace(self):
    tmpl = {'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'AResource': {'Type': 'ResourceWithPropsType', 'Properties': {'Foo': 'foo'}}}}
    self.stack = stack.Stack(self.ctx, 'update_test_stack', template.Template(tmpl), disable_rollback=False)
    self.stack.store()
    self.stack.create()
    self.assertEqual((stack.Stack.CREATE, stack.Stack.COMPLETE), self.stack.state)
    tmpl2 = {'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'AResource': {'Type': 'ResourceWithPropsType', 'Properties': {'Foo': 'bar'}}}}
    updated_stack = stack.Stack(self.ctx, 'updated_stack', template.Template(tmpl2), disable_rollback=False)
    mock_delete = self.patchobject(generic_rsrc.ResourceWithProps, 'handle_delete', side_effect=[Exception, None, None])
    self.stack.update(updated_stack)
    self.assertEqual((stack.Stack.ROLLBACK, stack.Stack.COMPLETE), self.stack.state)
    self.assertEqual(3, mock_delete.call_count)