from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import keystone as k_plugin
from heat.engine import rsrc_defn
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_quota_handle_create(self):
    self.my_quota.physical_resource_name = mock.MagicMock(return_value='some_resource_id')
    self.my_quota.reparse()
    self.my_quota.handle_create()
    self.quotas.update.assert_called_once_with('some_project_id', healthmonitor=5, listener=5, loadbalancer=5, pool=5, member=5)
    self.assertEqual('some_resource_id', self.my_quota.resource_id)