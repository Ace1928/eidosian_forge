import argparse
import copy
from random import choice
from random import randint
from unittest import mock
import uuid
from openstack.network.v2 import _proxy
from openstack.network.v2 import address_group as _address_group
from openstack.network.v2 import address_scope as _address_scope
from openstack.network.v2 import agent as network_agent
from openstack.network.v2 import auto_allocated_topology as allocated_topology
from openstack.network.v2 import availability_zone as _availability_zone
from openstack.network.v2 import extension as _extension
from openstack.network.v2 import flavor as _flavor
from openstack.network.v2 import local_ip as _local_ip
from openstack.network.v2 import local_ip_association as _local_ip_association
from openstack.network.v2 import ndp_proxy as _ndp_proxy
from openstack.network.v2 import network as _network
from openstack.network.v2 import network_ip_availability as _ip_availability
from openstack.network.v2 import network_segment_range as _segment_range
from openstack.network.v2 import port as _port
from openstack.network.v2 import rbac_policy as network_rbac
from openstack.network.v2 import segment as _segment
from openstack.network.v2 import service_profile as _flavor_profile
from openstack.network.v2 import trunk as _trunk
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit import utils
class FakeNetworkQosRuleType(object):
    """Fake one or more Network QoS rule types."""

    @staticmethod
    def create_one_qos_rule_type(attrs=None):
        """Create a fake Network QoS rule type.

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object with name, id, etc.
        """
        attrs = attrs or {}
        qos_rule_type_attrs = {'type': 'rule-type-' + uuid.uuid4().hex, 'location': 'MUNCHMUNCHMUNCH'}
        qos_rule_type_attrs.update(attrs)
        return fakes.FakeResource(info=copy.deepcopy(qos_rule_type_attrs), loaded=True)

    @staticmethod
    def create_qos_rule_types(attrs=None, count=2):
        """Create multiple fake Network QoS rule types.

        :param Dictionary attrs:
            A dictionary with all attributes
        :param int count:
            The number of QoS rule types to fake
        :return:
            A list of FakeResource objects faking the QoS rule types
        """
        qos_rule_types = []
        for i in range(0, count):
            qos_rule_types.append(FakeNetworkQosRuleType.create_one_qos_rule_type(attrs))
        return qos_rule_types