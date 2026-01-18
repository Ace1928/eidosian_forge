from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import node
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
def test_remove_trait_missing(self):
    self.session.delete.return_value.status_code = 400
    self.assertFalse(self.node.remove_trait(self.session, 'CUSTOM_MISSING'))
    self.session.delete.assert_called_once_with('nodes/%s/traits/%s' % (self.node.id, 'CUSTOM_MISSING'), headers=mock.ANY, microversion='1.37', retriable_status_codes=_common.RETRIABLE_STATUS_CODES)