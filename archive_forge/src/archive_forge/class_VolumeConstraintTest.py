from unittest import mock
import uuid
from cinderclient import exceptions as cinder_exc
from keystoneauth1 import exceptions as ks_exceptions
from heat.common import exception
from heat.engine.clients.os import cinder
from heat.tests import common
from heat.tests import utils
class VolumeConstraintTest(common.HeatTestCase):

    def setUp(self):
        super(VolumeConstraintTest, self).setUp()
        self.ctx = utils.dummy_context()
        self.mock_get_volume = mock.Mock()
        self.ctx.clients.client_plugin('cinder').get_volume = self.mock_get_volume
        self.constraint = cinder.VolumeConstraint()

    def test_validation(self):
        self.mock_get_volume.return_value = None
        self.assertTrue(self.constraint.validate('foo', self.ctx))

    def test_validation_error(self):
        self.mock_get_volume.side_effect = exception.EntityNotFound(entity='Volume', name='bar')
        self.assertFalse(self.constraint.validate('bar', self.ctx))