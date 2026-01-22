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
def _create_client_and_fixture(self):
    placement_client = place_client.PlacementAPIClient(mock.Mock())
    placement_fixture = self.useFixture(fixture.PlacementAPIClientFixture(placement_client))
    return (placement_client, placement_fixture)