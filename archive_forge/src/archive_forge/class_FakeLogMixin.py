import os
import sys
import fixtures
from oslo_config import cfg
from oslo_log import log as logging
import testscenarios
import testtools
from heat.common import context
from heat.common import messaging
from heat.common import policy
from heat.engine.clients.os import barbican
from heat.engine.clients.os import cinder
from heat.engine.clients.os import glance
from heat.engine.clients.os import keystone
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine.clients.os.keystone import keystone_constraints as ks_constr
from heat.engine.clients.os.neutron import neutron_constraints as neutron
from heat.engine.clients.os import nova
from heat.engine.clients.os import sahara
from heat.engine.clients.os import trove
from heat.engine import environment
from heat.engine import resource
from heat.engine import resources
from heat.engine import scheduler
from heat.tests import fakes
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
class FakeLogMixin(object):

    def setup_logging(self, quieten=True):
        default_level = logging.INFO
        if os.environ.get('OS_DEBUG') in _TRUE_VALUES:
            default_level = logging.DEBUG
        self.LOG = self.useFixture(fixtures.FakeLogger(level=default_level, format=_LOG_FORMAT))
        base_list = set([nlog.split('.')[0] for nlog in logging.logging.Logger.manager.loggerDict])
        for base in base_list:
            if base in TEST_DEFAULT_LOGLEVELS:
                self.useFixture(fixtures.FakeLogger(level=TEST_DEFAULT_LOGLEVELS[base], name=base, format=_LOG_FORMAT))
            elif base != 'heat':
                self.useFixture(fixtures.FakeLogger(name=base, format=_LOG_FORMAT))
        if quieten:
            for ll in TEST_DEFAULT_LOGLEVELS:
                if ll.startswith('heat.'):
                    self.useFixture(fixtures.FakeLogger(level=TEST_DEFAULT_LOGLEVELS[ll], name=ll, format=_LOG_FORMAT))