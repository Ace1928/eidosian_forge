from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import node
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
def test_set_via_update(self):
    self.node.is_maintenance = True
    self.node.commit(self.session)
    self.session.put.assert_called_once_with('nodes/%s/maintenance' % self.node.id, json={'reason': None}, headers=mock.ANY, microversion=mock.ANY)
    self.assertFalse(self.session.patch.called)