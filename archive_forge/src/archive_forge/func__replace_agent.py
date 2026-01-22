from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine.resources.openstack.neutron import subnet
from heat.engine import support
from heat.engine import translation
def _replace_agent(self, l3_agent_ids=None):
    ret = self.client().list_l3_agent_hosting_routers(self.resource_id)
    for agent in ret['agents']:
        self.client().remove_router_from_l3_agent(agent['id'], self.resource_id)
    if l3_agent_ids:
        for l3_agent_id in l3_agent_ids:
            self.client().add_router_to_l3_agent(l3_agent_id, {'router_id': self.resource_id})