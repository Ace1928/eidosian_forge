import copy
from unittest import mock
from neutronclient.common import exceptions as qe
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.resources.openstack.neutron import router
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_update_router_gateway_as_property(self):
    t, stack = self._test_router_with_gateway(for_update=True)
    rsrc = self.create_router(t, stack, 'router')
    self._assert_mock_call_create_with_router_gw()
    gateway_info = rsrc.FnGetAtt('external_gateway_info')
    self.assertEqual('fc68ea2c-b60b-4b4f-bd82-94ec81110766', gateway_info.get('network_id'))
    self.assertTrue(gateway_info.get('enable_snat'))
    props = t['resources']['router']['properties'].copy()
    props['external_gateway_info'] = {'network': 'other_public', 'enable_snat': False}
    update_template = rsrc.t.freeze(properties=props)

    def find_rsrc_for_update(resource, name_or_id, cmd_resource=None):
        id_mapping = {'subnet': 'sub1234', 'network': '91e47a57-7508-46fe-afc9-fc454e8580e1'}
        return id_mapping.get(resource)
    self.find_rsrc_mock.side_effect = find_rsrc_for_update
    scheduler.TaskRunner(rsrc.update, update_template)()
    self.assertEqual((rsrc.UPDATE, rsrc.COMPLETE), rsrc.state)
    self.update_mock.assert_called_with('3e46229d-8fce-4733-819a-b5fe630550f8', {'router': {'external_gateway_info': {'network_id': '91e47a57-7508-46fe-afc9-fc454e8580e1', 'enable_snat': False}}})
    gateway_info = rsrc.FnGetAtt('external_gateway_info')
    self.assertEqual('91e47a57-7508-46fe-afc9-fc454e8580e1', gateway_info.get('network_id'))
    self.assertFalse(gateway_info.get('enable_snat'))