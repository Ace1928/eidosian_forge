import copy
from unittest import mock
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources.openstack.keystone import service
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_service_handle_update_default_name(self):
    rsrc = self._setup_service_resource('test_update_default_name')
    rsrc.resource_id = '477e8273-60a7-4c41-b683-fdb0bc7cd151'
    rsrc.physical_resource_name = mock.MagicMock()
    rsrc.physical_resource_name.return_value = 'foo'
    prop_diff = {service.KeystoneService.NAME: None}
    rsrc.handle_update(json_snippet=None, tmpl_diff=None, prop_diff=prop_diff)
    self.services.update.assert_called_once_with(service=rsrc.resource_id, name='foo', type=None, description=None, enabled=None)