import copy
from unittest import mock
import warnings
import fixtures
from oslo_config import cfg
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import provision
from oslo_db.sqlalchemy import session
from oslo_messaging import conffixture
from neutron_lib.api import attributes
from neutron_lib.api import definitions
from neutron_lib.callbacks import manager
from neutron_lib.callbacks import registry
from neutron_lib.db import api as db_api
from neutron_lib.db import model_base
from neutron_lib.db import model_query
from neutron_lib.db import resource_extend
from neutron_lib.plugins import directory
from neutron_lib import rpc
from neutron_lib.tests.unit import fake_notifier
class APIDefinitionFixture(fixtures.Fixture):
    """Test fixture for testing neutron-lib API definitions.

    Extension API definition RESOURCE_ATTRIBUTE_MAP dicts get updated as
    part of standard extension processing/handling. While this behavior is
    fine for service runtime, it can impact testing scenarios whereby test1
    updates the attribute map (globally) and test2 doesn't expect the
    test1 updates.

    This fixture saves and restores 1 or more neutron-lib API definitions
    attribute maps. It should be used anywhere multiple tests can be run
    that might update an extension attribute map.

    In addition the fixture backs up and restores the global attribute
    RESOURCES base on the boolean value of its backup_global_resources
    attribute.
    """

    def __init__(self, *api_definitions):
        """Create a new instance.

        Consumers can also control if the fixture should handle the global
        attribute RESOURCE map using the backup_global_resources of the
        fixture instance. If True the fixture will also handle
        neutron_lib.api.attributes.RESOURCES.

        :param api_definitions: Zero or more API definitions the fixture
        should handle. If no api_definitions are passed, the default is
        to handle all neutron_lib API definitions as well as the global
        RESOURCES attribute map.
        """
        self.definitions = api_definitions or definitions._ALL_API_DEFINITIONS
        self._orig_attr_maps = {}
        self._orig_resources = {}
        self.backup_global_resources = True

    def _setUp(self):
        for api_def in self.definitions:
            self._orig_attr_maps[api_def.ALIAS] = (api_def, {k: copy.deepcopy(v) for k, v in api_def.RESOURCE_ATTRIBUTE_MAP.items()})
        if self.backup_global_resources:
            for resource, attrs in attributes.RESOURCES.items():
                self._orig_resources[resource] = copy.deepcopy(attrs)
        self.addCleanup(self._restore)

    def _restore(self):
        for alias, def_and_map in self._orig_attr_maps.items():
            api_def, attr_map = (def_and_map[0], def_and_map[1])
            api_def.RESOURCE_ATTRIBUTE_MAP.clear()
            api_def.RESOURCE_ATTRIBUTE_MAP.update(attr_map)
        if self.backup_global_resources:
            attributes.RESOURCES.clear()
            attributes.RESOURCES.update(self._orig_resources)

    @classmethod
    def all_api_definitions_fixture(cls):
        """Return a fixture that handles all neutron-lib api-defs."""
        return APIDefinitionFixture(*tuple(definitions._ALL_API_DEFINITIONS))