from neutron_lib.api.definitions import qos_default
from neutron_lib.tests.unit.api.definitions import base
from neutron_lib.tests.unit.api.definitions import test_qos
class QoSDefaultTestCase(base.DefinitionBaseTestCase):
    extension_module = qos_default
    extension_resources = test_qos.QoSDefinitionTestCase.extension_resources
    extension_attributes = ()