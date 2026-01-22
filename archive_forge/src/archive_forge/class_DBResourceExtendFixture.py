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
class DBResourceExtendFixture(fixtures.Fixture):

    def __init__(self, extended_functions=None):
        self.extended_functions = extended_functions or {}

    def _setUp(self):
        self._backup = copy.deepcopy(resource_extend._resource_extend_functions)
        resource_extend._resource_extend_functions = self.extended_functions
        self.addCleanup(self._restore)

    def _restore(self):
        resource_extend._resource_extend_functions = self._backup