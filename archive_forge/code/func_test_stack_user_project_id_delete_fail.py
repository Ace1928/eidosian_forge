import copy
import time
from unittest import mock
import fixtures
from keystoneauth1 import exceptions as kc_exceptions
from oslo_log import log as logging
from heat.common import exception
from heat.common import template_format
from heat.common import timeutils
from heat.engine.clients.os import keystone
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine.clients.os.keystone import heat_keystoneclient as hkc
from heat.engine import scheduler
from heat.engine import stack
from heat.engine import template
from heat.objects import snapshot as snapshot_object
from heat.objects import stack as stack_object
from heat.objects import user_creds as ucreds_object
from heat.tests import common
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
def test_stack_user_project_id_delete_fail(self):

    class FakeKeystoneClientFail(fake_ks.FakeKeystoneClient):

        def delete_stack_domain_project(self, project_id):
            raise kc_exceptions.Forbidden('Denied!')
    mock_kcp = self.patchobject(keystone.KeystoneClientPlugin, '_create', return_value=FakeKeystoneClientFail())
    self.stack = stack.Stack(self.ctx, 'user_project_init', self.tmpl, stack_user_project_id='aproject1234')
    self.stack.store()
    self.assertEqual('aproject1234', self.stack.stack_user_project_id)
    db_stack = stack_object.Stack.get_by_id(self.ctx, self.stack.id)
    self.assertEqual('aproject1234', db_stack.stack_user_project_id)
    self.stack.delete()
    mock_kcp.assert_called_with()
    self.assertEqual((stack.Stack.DELETE, stack.Stack.FAILED), self.stack.state)
    self.assertIn('Error deleting project', self.stack.status_reason)