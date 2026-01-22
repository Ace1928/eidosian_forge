from unittest import mock
from neutron_lib.api.definitions import portbindings
from neutron_lib import constants
from neutron_lib.services.qos import base as qos_base
from neutron_lib.services.qos import constants as qos_consts
from neutron_lib.tests import _base
def _make_driver(name='fake-driver', vif_types=[portbindings.VIF_TYPE_OVS], vnic_types=[portbindings.VNIC_NORMAL], supported_rules=SUPPORTED_RULES, requires_rpc_notifications=False):
    return qos_base.DriverBase(name, vif_types, vnic_types, supported_rules, requires_rpc_notifications=requires_rpc_notifications)