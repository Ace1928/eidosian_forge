from unittest import mock
from openstack import exceptions
from heat.engine.clients.os import senlin as senlin_plugin
from heat.tests import common
from heat.tests import utils
def test_validate_false(self):
    self.assertFalse(self.constraint.validate('Invalid_type', self.ctx))