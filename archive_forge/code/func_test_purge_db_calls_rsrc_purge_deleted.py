from datetime import datetime
from datetime import timedelta
from unittest import mock
from oslo_config import cfg
from heat.common import template_format
from heat.engine import environment
from heat.engine import stack as parser
from heat.engine import template as templatem
from heat.objects import raw_template as raw_template_object
from heat.objects import resource as resource_objects
from heat.objects import snapshot as snapshot_objects
from heat.objects import stack as stack_object
from heat.objects import sync_point as sync_point_object
from heat.rpc import worker_client
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
@mock.patch.object(resource_objects.Resource, 'purge_deleted')
def test_purge_db_calls_rsrc_purge_deleted(self, mock_rsrc_purge_delete, mock_cr):
    stack = tools.get_stack('test_stack', utils.dummy_context(), template=tools.string_template_five, convergence=True)
    stack.store()
    stack.purge_db()
    self.assertTrue(mock_rsrc_purge_delete.called)