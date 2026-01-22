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
class PluginDirectoryFixture(fixtures.Fixture):

    def __init__(self, plugin_directory=None):
        super().__init__()
        self.plugin_directory = plugin_directory or directory._PluginDirectory()

    def _setUp(self):
        self._orig_directory = directory._PLUGIN_DIRECTORY
        directory._PLUGIN_DIRECTORY = self.plugin_directory
        self.addCleanup(self._restore)

    def _restore(self):
        directory._PLUGIN_DIRECTORY = self._orig_directory