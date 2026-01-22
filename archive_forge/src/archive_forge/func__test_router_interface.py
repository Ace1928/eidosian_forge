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
def _test_router_interface(self, resolve_router=True):
    self.remove_if_mock.side_effect = [None, qe.NeutronClientException(status_code=404)]
    t = template_format.parse(neutron_template)
    stack = utils.parse_stack(t)
    self.stub_SubnetConstraint_validate()
    self.stub_RouterConstraint_validate()

    def find_rsrc(resource, name_or_id, cmd_resource=None):
        id_mapping = {'subnet': '91e47a57-7508-46fe-afc9-fc454e8580e1', 'router': '3e46229d-8fce-4733-819a-b5fe630550f8'}
        return id_mapping.get(resource)
    self.find_rsrc_mock.side_effect = find_rsrc
    router_key = 'router'
    subnet_key = 'subnet'
    rsrc = self.create_router_interface(t, stack, 'router_interface', properties={router_key: '3e46229d-8fce-4733-819a-b5fe630550f8', subnet_key: '91e47a57-7508-46fe-afc9-fc454e8580e1'})
    self.add_if_mock.assert_called_with('3e46229d-8fce-4733-819a-b5fe630550f8', {'subnet_id': '91e47a57-7508-46fe-afc9-fc454e8580e1'})
    if not resolve_router:
        self.assertEqual('3e46229d-8fce-4733-819a-b5fe630550f8', rsrc.properties.get(rsrc.ROUTER))
        self.assertIsNone(rsrc.properties.get(rsrc.ROUTER_ID))
    scheduler.TaskRunner(rsrc.delete)()
    rsrc.state_set(rsrc.CREATE, rsrc.COMPLETE, 'to delete again')
    scheduler.TaskRunner(rsrc.delete)()