import argparse
from unittest import mock
from osc_lib.tests import utils
from oslo_utils import uuidutils
from openstack.network.v2 import sfc_flow_classifier as flow_classifier
from openstack.network.v2 import sfc_port_chain as port_chain
from openstack.network.v2 import sfc_port_pair as port_pair
from openstack.network.v2 import sfc_port_pair_group as port_pair_group
from openstack.network.v2 import sfc_service_graph as service_graph
class TestNeutronClientOSCV2(utils.TestCommand):

    def setUp(self):
        super(TestNeutronClientOSCV2, self).setUp()
        self.namespace = argparse.Namespace()
        self.app.client_manager.session = mock.Mock()
        self.app.client_manager.network = mock.Mock()
        self.network = self.app.client_manager.network
        self.network.find_sfc_flow_classifier = mock.Mock(side_effect=lambda name_or_id, ignore_missing=False: {'id': name_or_id})
        self.network.find_sfc_port_chain = mock.Mock(side_effect=lambda name_or_id, ignore_missing=False: {'id': name_or_id})
        self.network.find_sfc_port_pair = mock.Mock(side_effect=lambda name_or_id, ignore_missing=False: {'id': name_or_id})
        self.network.find_sfc_port_pair_group = mock.Mock(side_effect=lambda name_or_id, ignore_missing=False: {'id': name_or_id})
        self.network.find_sfc_service_graph = mock.Mock(side_effect=lambda name_or_id, ignore_missing=False: {'id': name_or_id})
        self.network.find_port = mock.Mock(side_effect=lambda name_or_id, ignore_missing=False: {'id': name_or_id})