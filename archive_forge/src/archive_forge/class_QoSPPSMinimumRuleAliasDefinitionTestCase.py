from neutron_lib.api.definitions import qos_pps_minimum_rule_alias as apidef
from neutron_lib.services.qos import constants as qos_constants
from neutron_lib.tests.unit.api.definitions import base
class QoSPPSMinimumRuleAliasDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = apidef
    extension_resources = (apidef.MIN_PACKET_RATE_RULES_ALIAS,)
    extension_attributes = (qos_constants.MIN_KPPS, qos_constants.DIRECTION)