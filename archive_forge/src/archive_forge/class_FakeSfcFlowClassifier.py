import argparse
from unittest import mock
from osc_lib.tests import utils
from oslo_utils import uuidutils
from openstack.network.v2 import sfc_flow_classifier as flow_classifier
from openstack.network.v2 import sfc_port_chain as port_chain
from openstack.network.v2 import sfc_port_pair as port_pair
from openstack.network.v2 import sfc_port_pair_group as port_pair_group
from openstack.network.v2 import sfc_service_graph as service_graph
class FakeSfcFlowClassifier(object):
    """Fake flow classifier attributes."""

    @staticmethod
    def create_flow_classifier(attrs=None):
        """Create a fake flow classifier.

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A Dictionary with faking port chain attributes
        """
        attrs = attrs or {}
        flow_classifier_attrs = {'id': uuidutils.generate_uuid(), 'destination_ip_prefix': '2.2.2.2/32', 'destination_port_range_max': '90', 'destination_port_range_min': '80', 'ethertype': 'IPv4', 'logical_destination_port': uuidutils.generate_uuid(), 'logical_source_port': uuidutils.generate_uuid(), 'name': 'flow-classifier-name', 'description': 'fc_description', 'protocol': 'tcp', 'source_ip_prefix': '1.1.1.1/32', 'source_port_range_max': '20', 'source_port_range_min': '10', 'project_id': uuidutils.generate_uuid(), 'l7_parameters': {}}
        flow_classifier_attrs.update(attrs)
        return flow_classifier.SfcFlowClassifier(**flow_classifier_attrs)

    @staticmethod
    def create_flow_classifiers(attrs=None, count=1):
        """Create multiple flow classifiers.

        :param Dictionary attrs:
            A dictionary with all attributes
        :param int count:
            The number of flow classifiers to fake
        :return:
            A list of dictionaries faking the flow classifiers
        """
        flow_classifiers = []
        for _ in range(count):
            flow_classifiers.append(FakeSfcFlowClassifier.create_flow_classifier(attrs))
        return flow_classifiers