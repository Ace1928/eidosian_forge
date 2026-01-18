from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.tests.unit import base
Test that we detect invalid arguments passed to update_port