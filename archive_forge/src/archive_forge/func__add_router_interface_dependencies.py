from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine.resources.openstack.neutron import port
from heat.engine.resources.openstack.neutron import router
from heat.engine import support
from heat.engine import translation
def _add_router_interface_dependencies(self, deps, resource):

    def port_on_subnet(resource, subnet):
        if not resource.has_interface('OS::Neutron::Port'):
            return False
        try:
            fixed_ips = resource.properties.get(port.Port.FIXED_IPS)
        except (ValueError, TypeError):
            return False
        if not fixed_ips:
            if subnet is None:
                return True
            try:
                p_net = resource.properties.get(port.Port.NETWORK) or resource.properties.get(port.Port.NETWORK_ID)
            except (ValueError, TypeError):
                return False
            if p_net:
                try:
                    network = self.client().show_network(p_net)['network']
                    return subnet in network['subnets']
                except Exception as exc:
                    LOG.info('Ignoring Neutron error while getting FloatingIP dependencies: %s', str(exc))
                    return False
        else:
            try:
                fixed_ips = resource.properties.get(port.Port.FIXED_IPS)
            except (ValueError, TypeError):
                return False
            for fixed_ip in fixed_ips:
                port_subnet = fixed_ip.get(port.Port.FIXED_IP_SUBNET) or fixed_ip.get(port.Port.FIXED_IP_SUBNET_ID)
                if subnet == port_subnet:
                    return True
        return False
    interface_subnet = resource.properties.get(router.RouterInterface.SUBNET) or resource.properties.get(router.RouterInterface.SUBNET_ID)
    for d in deps.graph()[self]:
        if port_on_subnet(d, interface_subnet):
            deps += (self, resource)
            break