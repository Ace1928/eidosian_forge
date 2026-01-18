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
def test_should_skip_resource_cleanup(self):
    excluded = ['block_storage.backup']
    self.assertTrue(self.sot.should_skip_resource_cleanup('backup', excluded))
    self.assertFalse(self.sot.should_skip_resource_cleanup('volume', excluded))