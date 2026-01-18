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
def test_service_handle_update(self):
    rsrc = self._setup_service_resource('test_update')
    rsrc.resource_id = '477e8273-60a7-4c41-b683-fdb0bc7cd151'
    prop_diff = {service.KeystoneService.NAME: 'test_service_1_updated', service.KeystoneService.DESCRIPTION: 'Test Service updated', service.KeystoneService.TYPE: 'heat_updated', service.KeystoneService.ENABLED: False}
    rsrc.handle_update(json_snippet=None, tmpl_diff=None, prop_diff=prop_diff)
    self.services.update.assert_called_once_with(service=rsrc.resource_id, name=prop_diff[service.KeystoneService.NAME], description=prop_diff[service.KeystoneService.DESCRIPTION], type=prop_diff[service.KeystoneService.TYPE], enabled=prop_diff[service.KeystoneService.ENABLED])