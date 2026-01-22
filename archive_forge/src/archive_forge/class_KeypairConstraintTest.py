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
class KeypairConstraintTest(common.HeatTestCase):

    def test_validation(self):
        client = fakes_nova.FakeClient()
        self.patchobject(nova.NovaClientPlugin, 'get_max_microversion', return_value='2.27')
        self.patchobject(nova.NovaClientPlugin, '_create', return_value=client)
        client.keypairs = mock.MagicMock()
        key = collections.namedtuple('Key', ['name'])
        key.name = 'foo'
        client.keypairs.get.side_effect = [fakes_nova.fake_exception(), key]
        constraint = nova.KeypairConstraint()
        ctx = utils.dummy_context()
        self.assertFalse(constraint.validate('bar', ctx))
        self.assertTrue(constraint.validate('foo', ctx))
        self.assertTrue(constraint.validate('', ctx))
        nova.NovaClientPlugin._create.assert_called_once_with(version='2.27')
        calls = [mock.call('bar'), mock.call(key.name)]
        client.keypairs.get.assert_has_calls(calls)