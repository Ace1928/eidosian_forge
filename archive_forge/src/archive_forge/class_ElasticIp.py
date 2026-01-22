from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine.clients import client_exception
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources.aws.ec2 import internet_gateway
from heat.engine.resources.aws.ec2 import vpc
from heat.engine import support
class ElasticIp(resource.Resource):
    PROPERTIES = DOMAIN, INSTANCE_ID = ('Domain', 'InstanceId')
    ATTRIBUTES = ALLOCATION_ID, = ('AllocationId',)
    properties_schema = {DOMAIN: properties.Schema(properties.Schema.STRING, _('Set to "vpc" to have IP address allocation associated to your VPC.'), support_status=support.SupportStatus(status=support.DEPRECATED, message=_('Now we only allow vpc here, so no need to set up this tag anymore.'), version='9.0.0'), constraints=[constraints.AllowedValues(['vpc'])]), INSTANCE_ID: properties.Schema(properties.Schema.STRING, _('Instance ID to associate with EIP.'), update_allowed=True, constraints=[constraints.CustomConstraint('nova.server')])}
    attributes_schema = {ALLOCATION_ID: attributes.Schema(_('ID that AWS assigns to represent the allocation of the address for use with Amazon VPC. Returned only for VPC elastic IP addresses.'), type=attributes.Schema.STRING)}
    default_client_name = 'nova'

    def __init__(self, name, json_snippet, stack):
        super(ElasticIp, self).__init__(name, json_snippet, stack)
        self.ipaddress = None

    def _ipaddress(self):
        if self.ipaddress is None and self.resource_id is not None:
            try:
                ips = self.neutron().show_floatingip(self.resource_id)
            except Exception as ex:
                self.client_plugin('neutron').ignore_not_found(ex)
            else:
                self.ipaddress = ips['floatingip']['floating_ip_address']
        return self.ipaddress or ''

    def handle_create(self):
        """Allocate a floating IP for the current tenant."""
        ips = None
        ext_net = internet_gateway.InternetGateway.get_external_network_id(self.neutron())
        props = {'floating_network_id': ext_net}
        ips = self.neutron().create_floatingip({'floatingip': props})['floatingip']
        self.resource_id_set(ips['id'])
        self.ipaddress = ips['floating_ip_address']
        LOG.info('ElasticIp create %s', str(ips))
        instance_id = self.properties[self.INSTANCE_ID]
        if instance_id:
            self.client_plugin().associate_floatingip(instance_id, ips['id'])

    def handle_delete(self):
        if self.resource_id is None:
            return
        instance_id = self.properties[self.INSTANCE_ID]
        if instance_id:
            with self.client_plugin().ignore_not_found:
                self.client_plugin().dissociate_floatingip(self.resource_id)
        with self.client_plugin('neutron').ignore_not_found:
            self.neutron().delete_floatingip(self.resource_id)

    def handle_update(self, json_snippet, tmpl_diff, prop_diff):
        if prop_diff:
            if self.INSTANCE_ID in prop_diff:
                instance_id = prop_diff[self.INSTANCE_ID]
                if instance_id:
                    self.client_plugin().associate_floatingip(instance_id, self.resource_id)
                else:
                    self.client_plugin().dissociate_floatingip(self.resource_id)

    def get_reference_id(self):
        eip = self._ipaddress()
        if eip:
            return str(eip)
        else:
            return str(self.name)

    def _resolve_attribute(self, name):
        if name == self.ALLOCATION_ID:
            return str(self.resource_id)