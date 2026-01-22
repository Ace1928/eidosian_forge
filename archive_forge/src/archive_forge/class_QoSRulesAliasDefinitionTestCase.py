from neutron_lib.api.definitions import qos_rules_alias
from neutron_lib.services.qos import constants as q_const
from neutron_lib.tests.unit.api.definitions import base
class QoSRulesAliasDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = qos_rules_alias
    extension_resources = (qos_rules_alias.BANDWIDTH_LIMIT_RULES_ALIAS, qos_rules_alias.DSCP_MARKING_RULES_ALIAS, qos_rules_alias.MIN_BANDWIDTH_RULES_ALIAS)
    extension_attributes = (q_const.DIRECTION, q_const.MAX_BURST, q_const.DSCP_MARK, q_const.MIN_KBPS, q_const.MAX_KBPS)