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
class DBAPIContextManagerFixture(fixtures.Fixture):

    def __init__(self, mock_context_manager=mock.ANY):
        self.cxt_manager = mock.Mock() if mock_context_manager == mock.ANY else mock_context_manager
        self._backup_mgr = None

    def _setUp(self):
        self._backup_mgr = db_api._CTX_MANAGER
        db_api._CTX_MANAGER = self.cxt_manager
        self.addCleanup(self._restore)

    def _restore(self):
        db_api._CTX_MANAGER = self._backup_mgr