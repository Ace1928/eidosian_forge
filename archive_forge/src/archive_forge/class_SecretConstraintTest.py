import collections
from unittest import mock
from barbicanclient import exceptions
from heat.common import exception
from heat.engine.clients.os import barbican
from heat.tests import common
from heat.tests import utils
class SecretConstraintTest(common.HeatTestCase):

    def setUp(self):
        super(SecretConstraintTest, self).setUp()
        self.ctx = utils.dummy_context()
        self.mock_get_secret_by_ref = mock.Mock()
        self.ctx.clients.client_plugin('barbican').get_secret_by_ref = self.mock_get_secret_by_ref
        self.constraint = barbican.SecretConstraint()

    def test_validation(self):
        self.mock_get_secret_by_ref.return_value = {}
        self.assertTrue(self.constraint.validate('foo', self.ctx))

    def test_validation_error(self):
        self.mock_get_secret_by_ref.side_effect = exception.EntityNotFound(entity='Secret', name='bar')
        self.assertFalse(self.constraint.validate('bar', self.ctx))