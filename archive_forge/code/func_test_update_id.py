import copy
import queue
from unittest import mock
from keystoneauth1 import session
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack import exceptions
from openstack import proxy
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
def test_update_id(self):
    rv = self.sot._update(UpdateableResource, self.fake_id, **self.attrs)
    self.assertEqual(rv, self.fake_result)
    self.res.commit.assert_called_once_with(self.sot, base_path=None)