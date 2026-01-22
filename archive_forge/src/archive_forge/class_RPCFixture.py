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
class RPCFixture(fixtures.Fixture):

    def _setUp(self):
        mock.patch.object(rpc.Connection, 'consume_in_threads', return_value=[]).start()
        self.useFixture(fixtures.MonkeyPatch('oslo_messaging.Notifier', fake_notifier.FakeNotifier))
        self.messaging_conf = conffixture.ConfFixture(CONF)
        self.messaging_conf.transport_url = 'fake:/'
        self.messaging_conf.response_timeout = 0
        self.useFixture(self.messaging_conf)
        self.addCleanup(rpc.cleanup)
        rpc.init(CONF)