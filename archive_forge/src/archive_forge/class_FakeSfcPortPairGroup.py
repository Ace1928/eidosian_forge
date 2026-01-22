import argparse
from unittest import mock
from osc_lib.tests import utils
from oslo_utils import uuidutils
from openstack.network.v2 import sfc_flow_classifier as flow_classifier
from openstack.network.v2 import sfc_port_chain as port_chain
from openstack.network.v2 import sfc_port_pair as port_pair
from openstack.network.v2 import sfc_port_pair_group as port_pair_group
from openstack.network.v2 import sfc_service_graph as service_graph
class FakeSfcPortPairGroup(object):
    """Fake port pair group attributes."""

    @staticmethod
    def create_port_pair_group(attrs=None):
        """Create a fake port pair group.

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A Dictionary with id, name, description, port_pairs, group_id
            port_pair_group_parameters, project_id
        """
        attrs = attrs or {}
        port_pair_group_attrs = {'id': uuidutils.generate_uuid(), 'name': 'port-pair-group-name', 'description': 'description', 'port_pairs': uuidutils.generate_uuid(), 'port_pair_group_parameters': {'lb_fields': []}, 'project_id': uuidutils.generate_uuid(), 'tap_enabled': False}
        port_pair_group_attrs.update(attrs)
        return port_pair_group.SfcPortPairGroup(**port_pair_group_attrs)

    @staticmethod
    def create_port_pair_groups(attrs=None, count=1):
        """Create multiple port pair groups.

        :param Dictionary attrs:
            A dictionary with all attributes
        :param int count:
            The number of port_pair_groups to fake
        :return:
            A list of dictionaries faking the port pair groups
        """
        port_pair_groups = []
        for _ in range(count):
            port_pair_groups.append(FakeSfcPortPairGroup.create_port_pair_group(attrs))
        return port_pair_groups