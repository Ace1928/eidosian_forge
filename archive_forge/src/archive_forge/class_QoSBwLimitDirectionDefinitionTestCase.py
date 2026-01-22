from neutron_lib.api.definitions import qos
from neutron_lib.api.definitions import qos_bw_limit_direction
from neutron_lib.services.qos import constants as qos_const
from neutron_lib.tests.unit.api.definitions import base
class QoSBwLimitDirectionDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = qos_bw_limit_direction
    extension_subresources = (qos.BANDWIDTH_LIMIT_RULES,)
    extension_attributes = (qos_const.DIRECTION,)