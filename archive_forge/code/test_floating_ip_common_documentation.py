from unittest.mock import patch
from openstack.cloud import meta
from openstack.compute.v2 import server as _server
from openstack import connection
from openstack.tests import fakes
from openstack.tests.unit import base

test_floating_ip_common
----------------------------------

Tests floating IP resource methods for Neutron and Nova-network.
