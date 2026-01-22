from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine import support
class MeteringRule(neutron.NeutronResource):
    """A resource to create rule for some label.

    Resource for allowing specified label to measure the traffic for a specific
    set of ip range.
    """
    support_status = support.SupportStatus(version='2014.1')
    entity = 'metering_label_rule'
    PROPERTIES = METERING_LABEL_ID, REMOTE_IP_PREFIX, DIRECTION, EXCLUDED = ('metering_label_id', 'remote_ip_prefix', 'direction', 'excluded')
    ATTRIBUTES = DIRECTION_ATTR, EXCLUDED_ATTR, METERING_LABEL_ID_ATTR, REMOTE_IP_PREFIX_ATTR = ('direction', 'excluded', 'metering_label_id', 'remote_ip_prefix')
    properties_schema = {METERING_LABEL_ID: properties.Schema(properties.Schema.STRING, _('The metering label ID to associate with this metering rule.'), required=True), REMOTE_IP_PREFIX: properties.Schema(properties.Schema.STRING, _('Indicates remote IP prefix to be associated with this metering rule.'), required=True), DIRECTION: properties.Schema(properties.Schema.STRING, _('The direction in which metering rule is applied, either ingress or egress.'), default='ingress', constraints=[constraints.AllowedValues(('ingress', 'egress'))]), EXCLUDED: properties.Schema(properties.Schema.BOOLEAN, _('Specify whether the remote_ip_prefix will be excluded or not from traffic counters of the metering label. For example to not count the traffic of a specific IP address of a range.'), default='False')}
    attributes_schema = {DIRECTION_ATTR: attributes.Schema(_('The direction in which metering rule is applied.'), type=attributes.Schema.STRING), EXCLUDED_ATTR: attributes.Schema(_('Exclude state for cidr.'), type=attributes.Schema.STRING), METERING_LABEL_ID_ATTR: attributes.Schema(_('The metering label ID to associate with this metering rule.'), type=attributes.Schema.STRING), REMOTE_IP_PREFIX_ATTR: attributes.Schema(_('CIDR to be associated with this metering rule.'), type=attributes.Schema.STRING)}

    def handle_create(self):
        props = self.prepare_properties(self.properties, self.physical_resource_name())
        metering_label_rule = self.client().create_metering_label_rule({'metering_label_rule': props})['metering_label_rule']
        self.resource_id_set(metering_label_rule['id'])

    def handle_delete(self):
        try:
            self.client().delete_metering_label_rule(self.resource_id)
        except Exception as ex:
            self.client_plugin().ignore_not_found(ex)
        else:
            return True