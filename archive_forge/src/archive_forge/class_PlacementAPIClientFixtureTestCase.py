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
class PlacementAPIClientFixtureTestCase(base.BaseTestCase):

    def _create_client_and_fixture(self):
        placement_client = place_client.PlacementAPIClient(mock.Mock())
        placement_fixture = self.useFixture(fixture.PlacementAPIClientFixture(placement_client))
        return (placement_client, placement_fixture)

    def test_post(self):
        p_client, p_fixture = self._create_client_and_fixture()
        p_client.create_resource_provider('resource')
        p_fixture.mock_post.assert_called_once()

    def test_put(self):
        inventory = {'total': 42}
        p_client, p_fixture = self._create_client_and_fixture()
        p_client.update_resource_provider_inventory('resource', inventory, 'class_name', 1)
        p_fixture.mock_put.assert_called_once()

    def test_delete(self):
        p_client, p_fixture = self._create_client_and_fixture()
        p_client.delete_resource_provider('resource')
        p_fixture.mock_delete.assert_called_once()

    def test_get(self):
        p_client, p_fixture = self._create_client_and_fixture()
        p_client.list_aggregates('resource')
        p_fixture.mock_get.assert_called_once()