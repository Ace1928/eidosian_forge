from unittest import mock
from openstack import exceptions
from heat.engine.clients.os import senlin as senlin_plugin
from heat.tests import common
from heat.tests import utils
def test_validate_true(self):
    self.assertTrue(self.constraint.validate('senlin.policy.deletion-1.0', self.ctx))