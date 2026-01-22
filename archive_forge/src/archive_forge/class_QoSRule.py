from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine import support
class QoSRule(neutron.NeutronResource):
    """A resource for Neutron QoS base rule."""
    required_service_extension = 'qos'
    support_status = support.SupportStatus(version='6.0.0')
    PROPERTIES = POLICY, TENANT_ID = ('policy', 'tenant_id')
    properties_schema = {POLICY: properties.Schema(properties.Schema.STRING, _('ID or name of the QoS policy.'), required=True, constraints=[constraints.CustomConstraint('neutron.qos_policy')]), TENANT_ID: properties.Schema(properties.Schema.STRING, _('The owner tenant ID of this rule.'))}

    def __init__(self, name, json_snippet, stack):
        super(QoSRule, self).__init__(name, json_snippet, stack)
        self._policy_id = None

    @property
    def policy_id(self):
        if not self._policy_id:
            self._policy_id = self.client_plugin().get_qos_policy_id(self.properties[self.POLICY])
        return self._policy_id