import argparse
from unittest import mock
from osc_lib.tests import utils
from oslo_utils import uuidutils
from openstack.network.v2 import sfc_flow_classifier as flow_classifier
from openstack.network.v2 import sfc_port_chain as port_chain
from openstack.network.v2 import sfc_port_pair as port_pair
from openstack.network.v2 import sfc_port_pair_group as port_pair_group
from openstack.network.v2 import sfc_service_graph as service_graph
class FakeSfcPortPair(object):
    """Fake port pair attributes."""

    @staticmethod
    def create_port_pair(attrs=None):
        """Create a fake port pair.

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A Dictionary with id, name, description, ingress, egress,
            service-function-parameter, project_id
        """
        attrs = attrs or {}
        port_pair_attrs = {'description': 'description', 'egress': uuidutils.generate_uuid(), 'id': uuidutils.generate_uuid(), 'ingress': uuidutils.generate_uuid(), 'name': 'port-pair-name', 'service_function_parameters': [('correlation', None), ('weight', 1)], 'project_id': uuidutils.generate_uuid()}
        port_pair_attrs.update(attrs)
        return port_pair.SfcPortPair(**port_pair_attrs)

    @staticmethod
    def create_port_pairs(attrs=None, count=1):
        """Create multiple port_pairs.

        :param Dictionary attrs:
            A dictionary with all attributes
        :param int count:
            The number of port_pairs to fake
        :return:
            A list of dictionaries faking the port_pairs
        """
        port_pairs = []
        for _ in range(count):
            port_pairs.append(FakeSfcPortPair.create_port_pair(attrs))
        return port_pairs