from unittest import mock
from neutron_lib.api.definitions import portbindings
from neutron_lib import constants
from neutron_lib.services.qos import base as qos_base
from neutron_lib.services.qos import constants as qos_consts
from neutron_lib.tests import _base
class TestDriverBase(_base.BaseTestCase):

    def test_is_loaded(self):
        self.assertTrue(_make_driver().is_loaded())

    def test_is_vif_type_compatible(self):
        self.assertTrue(_make_driver().is_vif_type_compatible(portbindings.VIF_TYPE_OVS))
        self.assertFalse(_make_driver().is_vif_type_compatible(portbindings.VIF_TYPE_BRIDGE))

    def test_is_vnic_compatible(self):
        self.assertTrue(_make_driver().is_vnic_compatible(portbindings.VNIC_NORMAL))
        self.assertFalse(_make_driver().is_vnic_compatible(portbindings.VNIC_BAREMETAL))

    def test_is_rule_supported_with_unsupported_rule(self):
        self.assertFalse(_make_driver().is_rule_supported(_make_rule()))

    def test_is_rule_supported(self):
        self.assertTrue(_make_driver().is_rule_supported(_make_rule(rule_type=qos_consts.RULE_TYPE_MINIMUM_BANDWIDTH, params={qos_consts.MIN_KBPS: None, qos_consts.DIRECTION: constants.EGRESS_DIRECTION})))
        self.assertFalse(_make_driver().is_rule_supported(_make_rule(rule_type=qos_consts.RULE_TYPE_MINIMUM_BANDWIDTH, params={qos_consts.MIN_KBPS: None, qos_consts.DIRECTION: constants.INGRESS_DIRECTION})))