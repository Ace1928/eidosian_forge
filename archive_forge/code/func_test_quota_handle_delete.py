from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import keystone as k_plugin
from heat.engine import rsrc_defn
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_quota_handle_delete(self):
    self.my_quota.reparse()
    self.my_quota.handle_delete()
    self.quotas.delete.assert_called_once_with('some_project_id')