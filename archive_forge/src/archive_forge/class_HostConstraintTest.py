import collections
from unittest import mock
import uuid
from novaclient import client as nc
from novaclient import exceptions as nova_exceptions
from oslo_config import cfg
from oslo_serialization import jsonutils as json
import requests
from heat.common import exception
from heat.engine.clients.os import nova
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
class HostConstraintTest(common.HeatTestCase):

    def setUp(self):
        super(HostConstraintTest, self).setUp()
        self.ctx = utils.dummy_context()
        self.mock_get_host = mock.Mock()
        self.ctx.clients.client_plugin('nova').get_host = self.mock_get_host
        self.constraint = nova.HostConstraint()

    def test_validation(self):
        self.mock_get_host.return_value = mock.MagicMock()
        self.assertTrue(self.constraint.validate('foo', self.ctx))

    def test_validation_error(self):
        self.mock_get_host.side_effect = exception.EntityNotFound(entity='Host', name='bar')
        self.assertRaises(exception.EntityNotFound, self.constraint.validate, 'bar', self.ctx)