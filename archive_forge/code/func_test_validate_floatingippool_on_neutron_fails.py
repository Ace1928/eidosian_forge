from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.clients.os import nova
from heat.engine.clients.os import sahara
from heat.engine.resources.openstack.sahara import templates as st
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_validate_floatingippool_on_neutron_fails(self):
    ngt = self._init_ngt(self.t)
    self.patchobject(neutron.NeutronClientPlugin, 'find_resourceid_by_name_or_id').side_effect = [neutron.exceptions.NeutronClientNoUniqueMatch(message='Too many'), neutron.exceptions.NeutronClientException(message='Not found', status_code=404)]
    ex = self.assertRaises(exception.StackValidationFailed, ngt.validate)
    self.assertEqual('Too many', str(ex))
    ex = self.assertRaises(exception.StackValidationFailed, ngt.validate)
    self.assertEqual('Not found', str(ex))