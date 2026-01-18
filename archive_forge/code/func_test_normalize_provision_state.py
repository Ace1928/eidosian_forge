from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import node
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
def test_normalize_provision_state(self):
    attrs = dict(FAKE, provision_state=None)
    sot = node.Node(**attrs)
    self.assertEqual('available', sot.provision_state)