import re
from unittest import mock
from oslo_config import cfg
from oslo_db import options
from oslotest import base
from neutron_lib.api import attributes
from neutron_lib.api.definitions import port
from neutron_lib.callbacks import registry
from neutron_lib.db import model_base
from neutron_lib.db import resource_extend
from neutron_lib import fixture
from neutron_lib.placement import client as place_client
from neutron_lib.plugins import directory
from neutron_lib.tests.unit.api import test_attributes
class CallbackRegistryFixtureTestCase(base.BaseTestCase):

    def setUp(self):
        super(CallbackRegistryFixtureTestCase, self).setUp()
        self.manager = mock.Mock()
        self.useFixture(fixture.CallbackRegistryFixture(callback_manager=self.manager))

    def test_fixture(self):
        registry.publish('a', 'b', self, payload=mock.ANY)
        self.assertTrue(self.manager.publish.called)