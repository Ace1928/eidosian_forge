from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import keystone as k_plugin
from heat.engine import rsrc_defn
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_quota_handle_update(self):
    tmpl_diff = mock.MagicMock()
    prop_diff = mock.MagicMock()
    props = {'project': 'some_project_id', 'pool': 1, 'member': 2, 'listener': 3, 'loadbalancer': 4, 'healthmonitor': 2}
    json_snippet = rsrc_defn.ResourceDefinition(self.my_quota.name, 'OS::Octavia::Quota', properties=props)
    self.my_quota.reparse()
    self.my_quota.handle_update(json_snippet, tmpl_diff, prop_diff)
    self.quotas.update.assert_called_once_with('some_project_id', pool=1, member=2, listener=3, loadbalancer=4, healthmonitor=2)