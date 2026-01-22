from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine import support
class QoSDscpMarkingRule(QoSRule):
    """A resource for Neutron QoS DSCP marking rule.

    This rule can be associated with QoS policy, and then the policy
    can be used by neutron port and network, to provide DSCP marking
    QoS capabilities.

    The default policy usage of this resource is limited to
    administrators only.
    """
    support_status = support.SupportStatus(version='7.0.0')
    entity = 'dscp_marking_rule'
    PROPERTIES = DSCP_MARK, = ('dscp_mark',)
    properties_schema = {DSCP_MARK: properties.Schema(properties.Schema.INTEGER, _('DSCP mark between 0 and 56, except 2-6, 42, 44, and 50-54.'), required=True, update_allowed=True, constraints=[constraints.AllowedValues([0, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 46, 48, 56])])}
    properties_schema.update(QoSRule.properties_schema)

    def handle_create(self):
        props = self.prepare_properties(self.properties, self.physical_resource_name())
        props.pop(self.POLICY)
        rule = self.client().create_dscp_marking_rule(self.policy_id, {'dscp_marking_rule': props})['dscp_marking_rule']
        self.resource_id_set(rule['id'])

    def handle_delete(self):
        if self.resource_id is None:
            return
        with self.client_plugin().ignore_not_found:
            self.client().delete_dscp_marking_rule(self.resource_id, self.policy_id)

    def handle_update(self, json_snippet, tmpl_diff, prop_diff):
        if prop_diff:
            self.client().update_dscp_marking_rule(self.resource_id, self.policy_id, {'dscp_marking_rule': prop_diff})

    def _res_get_args(self):
        return [self.resource_id, self.policy_id]