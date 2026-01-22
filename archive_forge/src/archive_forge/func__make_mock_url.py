from copy import deepcopy
from unittest import mock
from openstack import exceptions
from openstack.network.v2.firewall_group import FirewallGroup
from openstack.network.v2.firewall_policy import FirewallPolicy
from openstack.network.v2.firewall_rule import FirewallRule
from openstack.tests.unit import base
def _make_mock_url(self, *args, **params):
    params_list = ['='.join([k, v]) for k, v in params.items()]
    return self.get_mock_url('network', 'public', append=['v2.0', 'fwaas'] + list(args), qs_elements=params_list or None)