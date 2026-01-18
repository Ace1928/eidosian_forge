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
def test_stack_delete_resourcefailure(self):
    tmpl = {'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'AResource': {'Type': 'GenericResourceType'}}}
    mock_rd = self.patchobject(generic_rsrc.GenericResource, 'handle_delete', side_effect=Exception('foo'))
    self.stack = stack.Stack(self.ctx, 'delete_test_fail', template.Template(tmpl))
    self.stack.store()
    self.stack.create()
    self.assertEqual((self.stack.CREATE, self.stack.COMPLETE), self.stack.state)
    self.stack.delete()
    mock_rd.assert_called_once_with()
    self.assertEqual((self.stack.DELETE, self.stack.FAILED), self.stack.state)
    self.assertEqual('Resource DELETE failed: Exception: resources.AResource: foo', self.stack.status_reason)