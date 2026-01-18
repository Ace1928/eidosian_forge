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
def test_update_failure_recovery(self):
    """Check that rollback still works with dynamic metadata.

        This test fails the second instance.
        """

    class ResourceTypeA(generic_rsrc.ResourceWithProps):
        count = 0

        def handle_create(self):
            ResourceTypeA.count += 1
            self.resource_id_set('%s%d' % (self.name, self.count))

        def handle_delete(self):
            return super(ResourceTypeA, self).handle_delete()
    resource._register_class('ResourceTypeA', ResourceTypeA)
    tmpl = {'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'AResource': {'Type': 'ResourceTypeA', 'Properties': {'Foo': 'abc'}}, 'BResource': {'Type': 'ResourceWithPropsType', 'Properties': {'Foo': {'Ref': 'AResource'}}}}}
    tmpl2 = {'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'AResource': {'Type': 'ResourceTypeA', 'Properties': {'Foo': 'smelly'}}, 'BResource': {'Type': 'ResourceWithPropsType', 'Properties': {'Foo': {'Ref': 'AResource'}}}}}
    self.stack = stack.Stack(self.ctx, 'update_test_stack', template.Template(tmpl), disable_rollback=True)
    self.stack.store()
    self.stack.create()
    self.assertEqual((stack.Stack.CREATE, stack.Stack.COMPLETE), self.stack.state)
    self.assertEqual('abc', self.stack['AResource']._stored_properties_data['Foo'])
    self.assertEqual('AResource1', self.stack['BResource']._stored_properties_data['Foo'])
    mock_create = self.patchobject(generic_rsrc.ResourceWithProps, 'handle_create', side_effect=[Exception, None])
    mock_delete = self.patchobject(generic_rsrc.ResourceWithProps, 'handle_delete')
    mock_delete_A = self.patchobject(ResourceTypeA, 'handle_delete')
    updated_stack = stack.Stack(self.ctx, 'updated_stack', template.Template(tmpl2), disable_rollback=True)
    self.stack.update(updated_stack)
    mock_create.assert_called_once_with()
    self.assertEqual((stack.Stack.UPDATE, stack.Stack.FAILED), self.stack.state)
    self.assertEqual('smelly', self.stack['AResource']._stored_properties_data['Foo'])
    self.stack = stack.Stack.load(self.ctx, self.stack.id)
    updated_stack2 = stack.Stack(self.ctx, 'updated_stack', template.Template(tmpl2), disable_rollback=True)
    self.stack.update(updated_stack2)
    self.assertEqual((stack.Stack.UPDATE, stack.Stack.COMPLETE), self.stack.state)
    self.assertEqual('smelly', self.stack['AResource']._stored_properties_data['Foo'])
    self.assertEqual('AResource2', self.stack['BResource']._stored_properties_data['Foo'])
    self.assertEqual(2, mock_create.call_count)
    self.assertEqual(2, mock_delete.call_count)
    mock_delete_A.assert_called_once_with()