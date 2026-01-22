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
class FloatingIPPortForward(neutron.NeutronResource):
    """A resource for creating port forwarding for floating IPs.

    This resource creates port forwarding for floating IPs.
    These are sub-resource of exsisting Floating ips, which requires the
    service_plugin and extension port_forwarding enabled and that the floating
    ip is not associated with a neutron port.
    """
    default_client_name = 'openstack'
    required_service_extension = 'floating-ip-port-forwarding'
    support_status = support.SupportStatus(status=support.SUPPORTED, version='19.0.0')
    PROPERTIES = INTERNAL_IP_ADDRESS, INTERNAL_PORT_NUMBER, EXTERNAL_PORT, INTERNAL_PORT, PROTOCOL, FLOATINGIP = ('internal_ip_address', 'internal_port_number', 'external_port', 'internal_port', 'protocol', 'floating_ip')
    properties_schema = {INTERNAL_IP_ADDRESS: properties.Schema(properties.Schema.STRING, _('Internal IP address to port forwarded to.'), required=True, update_allowed=True, constraints=[constraints.CustomConstraint('ip_addr')]), INTERNAL_PORT_NUMBER: properties.Schema(properties.Schema.INTEGER, _('Internal port number to port forward to.'), update_allowed=True, constraints=[constraints.Range(min=1, max=65535)]), EXTERNAL_PORT: properties.Schema(properties.Schema.INTEGER, _('External port address to port forward from.'), required=True, update_allowed=True, constraints=[constraints.Range(min=1, max=65535)]), INTERNAL_PORT: properties.Schema(properties.Schema.STRING, _('Name or ID of the internal_ip_address port.'), required=True, update_allowed=True, constraints=[constraints.CustomConstraint('neutron.port')]), PROTOCOL: properties.Schema(properties.Schema.STRING, _('Port protocol to forward.'), required=True, update_allowed=True, constraints=[constraints.AllowedValues(['tcp', 'udp', 'icmp', 'icmp6', 'sctp', 'dccp'])]), FLOATINGIP: properties.Schema(properties.Schema.STRING, _('Name or ID of the floating IP create port forwarding on.'), required=True)}

    def translation_rules(self, props):
        client_plugin = self.client_plugin()
        return [translation.TranslationRule(props, translation.TranslationRule.RESOLVE, [self.FLOATINGIP], client_plugin=client_plugin, finder='find_network_ip'), translation.TranslationRule(props, translation.TranslationRule.RESOLVE, [self.INTERNAL_PORT], client_plugin=client_plugin, finder='find_network_port')]

    def add_dependencies(self, deps):
        super(FloatingIPPortForward, self).add_dependencies(deps)
        for resource in self.stack.values():
            if resource.has_interface('OS::Neutron::RouterInterface'):

                def port_on_subnet(resource, subnet):
                    if not resource.has_interface('OS::Neutron::Port'):
                        return False
                    fixed_ips = resource.properties.get(port.Port.FIXED_IPS) or []
                    for fixed_ip in fixed_ips:
                        port_subnet = fixed_ip.get(port.Port.FIXED_IP_SUBNET) or fixed_ip.get(port.Port.FIXED_IP_SUBNET_ID)
                        return subnet == port_subnet
                    return False
                interface_subnet = resource.properties.get(router.RouterInterface.SUBNET) or resource.properties.get(router.RouterInterface.SUBNET_ID)
                for d in deps.graph()[self]:
                    if port_on_subnet(d, interface_subnet):
                        deps += (self, resource)
                        break

    def handle_create(self):
        props = self.prepare_properties(self.properties, self.name)
        fp = self.client().network.create_floating_ip_port_forwarding(props.pop(self.FLOATINGIP), **props)
        self.resource_id_set(fp.id)

    def handle_delete(self):
        if not self.resource_id:
            return
        self.client().network.delete_floating_ip_port_forwarding(self.properties[self.FLOATINGIP], self.resource_id, ignore_missing=True)

    def handle_check(self):
        self.client().network.get_port_forwarding(self.resource_id, self.properties[self.FLOATINGIP])

    def handle_update(self, prop_diff):
        if prop_diff:
            self.client().network.update_floating_ip_port_forwarding(self.properties[self.FLOATINGIP], self.resource_id, **prop_diff)