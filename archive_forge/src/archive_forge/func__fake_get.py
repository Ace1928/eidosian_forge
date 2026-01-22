from unittest import mock
from openstack.baremetal.v1 import _proxy
from openstack.baremetal.v1 import allocation
from openstack.baremetal.v1 import chassis
from openstack.baremetal.v1 import driver
from openstack.baremetal.v1 import node
from openstack.baremetal.v1 import port
from openstack.baremetal.v1 import port_group
from openstack.baremetal.v1 import volume_connector
from openstack.baremetal.v1 import volume_target
from openstack import exceptions
from openstack.tests.unit import base
from openstack.tests.unit import test_proxy_base
def _fake_get(_self, node):
    result = mock.Mock()
    result.id = getattr(node, 'id', node)
    if result.id == '1':
        result._check_state_reached.return_value = True
    elif result.id == '2':
        result._check_state_reached.side_effect = exceptions.ResourceFailure('boom')
    else:
        result._check_state_reached.return_value = False
    return result