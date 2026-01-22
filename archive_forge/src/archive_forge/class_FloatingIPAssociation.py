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
class FloatingIPAssociation(neutron.NeutronResource):
    """A resource for associating floating ips and ports.

    This resource allows associating a floating IP to a port with at least one
    IP address to associate with this floating IP.
    """
    PROPERTIES = FLOATINGIP_ID, PORT_ID, FIXED_IP_ADDRESS = ('floatingip_id', 'port_id', 'fixed_ip_address')
    properties_schema = {FLOATINGIP_ID: properties.Schema(properties.Schema.STRING, _('ID of the floating IP to associate.'), required=True, update_allowed=True), PORT_ID: properties.Schema(properties.Schema.STRING, _('ID of an existing port with at least one IP address to associate with this floating IP.'), required=True, update_allowed=True, constraints=[constraints.CustomConstraint('neutron.port')]), FIXED_IP_ADDRESS: properties.Schema(properties.Schema.STRING, _('IP address to use if the port has multiple addresses.'), update_allowed=True, constraints=[constraints.CustomConstraint('ip_addr')])}

    def add_dependencies(self, deps):
        super(FloatingIPAssociation, self).add_dependencies(deps)
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
        floatingip_id = props.pop(self.FLOATINGIP_ID)
        self.client().update_floatingip(floatingip_id, {'floatingip': props})
        self.resource_id_set(self.id)

    def handle_delete(self):
        if not self.resource_id:
            return
        with self.client_plugin().ignore_not_found:
            self.client().update_floatingip(self.properties[self.FLOATINGIP_ID], {'floatingip': {'port_id': None}})

    def handle_update(self, json_snippet, tmpl_diff, prop_diff):
        if prop_diff:
            floatingip_id = self.properties[self.FLOATINGIP_ID]
            port_id = self.properties[self.PORT_ID]
            if self.FLOATINGIP_ID in prop_diff:
                with self.client_plugin().ignore_not_found:
                    self.client().update_floatingip(floatingip_id, {'floatingip': {'port_id': None}})
            floatingip_id = prop_diff.get(self.FLOATINGIP_ID) or floatingip_id
            port_id = prop_diff.get(self.PORT_ID) or port_id
            fixed_ip_address = prop_diff.get(self.FIXED_IP_ADDRESS) or self.properties[self.FIXED_IP_ADDRESS]
            request_body = {'floatingip': {'port_id': port_id, 'fixed_ip_address': fixed_ip_address}}
            self.client().update_floatingip(floatingip_id, request_body)
            self.resource_id_set(self.id)