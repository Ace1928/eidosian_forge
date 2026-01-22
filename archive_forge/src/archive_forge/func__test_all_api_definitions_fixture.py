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
def _test_all_api_definitions_fixture(self, global_cleanup=True):
    apis = fixture.APIDefinitionFixture.all_api_definitions_fixture()
    apis.backup_global_resources = global_cleanup
    apis.setUp()
    asserteq = self.assertNotEqual if global_cleanup else self.assertEqual
    asserteq({}, apis._orig_resources)
    for r in test_attributes.TestCoreResources.CORE_DEFS:
        attributes.RESOURCES[r.COLLECTION_NAME]['_test_'] = {}
        r.RESOURCE_ATTRIBUTE_MAP['_test_'] = {}
    apis.cleanUp()
    for r in test_attributes.TestCoreResources.CORE_DEFS:
        self.assertNotIn('_test_', r.RESOURCE_ATTRIBUTE_MAP)
        global_assert = self.assertNotIn if global_cleanup else self.assertIn
        global_assert('_test_', attributes.RESOURCES[r.COLLECTION_NAME])
        if not global_cleanup:
            del attributes.RESOURCES[r.COLLECTION_NAME]['_test_']