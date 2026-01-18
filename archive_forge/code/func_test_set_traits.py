from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import node
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
def test_set_traits(self):
    traits = ['CUSTOM_FAKE', 'CUSTOM_REAL', 'CUSTOM_MISSING']
    self.node.set_traits(self.session, traits)
    self.session.put.assert_called_once_with('nodes/%s/traits' % self.node.id, json={'traits': ['CUSTOM_FAKE', 'CUSTOM_REAL', 'CUSTOM_MISSING']}, headers=mock.ANY, microversion='1.37', retriable_status_codes=_common.RETRIABLE_STATUS_CODES)