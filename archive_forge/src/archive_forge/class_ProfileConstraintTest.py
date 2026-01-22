from unittest import mock
from openstack import exceptions
from heat.engine.clients.os import senlin as senlin_plugin
from heat.tests import common
from heat.tests import utils
class ProfileConstraintTest(common.HeatTestCase):

    @mock.patch('openstack.connection.Connection')
    def setUp(self, mock_connection):
        super(ProfileConstraintTest, self).setUp()
        self.senlin_client = mock.MagicMock()
        self.ctx = utils.dummy_context()
        self.mock_get_profile = mock.Mock()
        self.ctx.clients.client('senlin').get_profile = self.mock_get_profile
        self.constraint = senlin_plugin.ProfileConstraint()

    def test_validate_true(self):
        self.mock_get_profile.return_value = None
        self.assertTrue(self.constraint.validate('PROFILE_ID', self.ctx))

    def test_validate_false(self):
        self.mock_get_profile.side_effect = exceptions.ResourceNotFound('PROFILE_ID')
        self.assertFalse(self.constraint.validate('PROFILE_ID', self.ctx))
        self.mock_get_profile.side_effect = exceptions.HttpException('PROFILE_ID')
        self.assertFalse(self.constraint.validate('PROFILE_ID', self.ctx))