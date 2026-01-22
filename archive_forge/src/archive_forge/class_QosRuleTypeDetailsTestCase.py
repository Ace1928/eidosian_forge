from neutron_lib.api.definitions import qos_rule_type_details
from neutron_lib.tests.unit.api.definitions import base
from neutron_lib.tests.unit.api.definitions import test_qos
class QosRuleTypeDetailsTestCase(base.DefinitionBaseTestCase):
    extension_module = qos_rule_type_details
    extension_resources = test_qos.QoSDefinitionTestCase.extension_resources
    extension_attributes = (qos_rule_type_details.DRIVERS,)