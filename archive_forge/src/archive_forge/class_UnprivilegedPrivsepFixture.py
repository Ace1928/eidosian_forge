import fixtures
import logging
import os
import sys
from oslo_config import fixture as cfg_fixture
from oslo_privsep import priv_context
class UnprivilegedPrivsepFixture(fixtures.Fixture):

    def __init__(self, context):
        self.context = context

    def setUp(self):
        super(UnprivilegedPrivsepFixture, self).setUp()
        self.conf = self.useFixture(cfg_fixture.Config()).conf
        self.conf.set_override('capabilities', [], group=self.context.cfg_section)
        for k in ('user', 'group'):
            self.conf.set_override(k, None, group=self.context.cfg_section)
        orig_pid = os.getpid()
        try:
            self.context.start(method=priv_context.Method.FORK)
        except Exception as e:
            if os.getpid() == orig_pid:
                raise
            LOG.exception(e)
            sys.exit(1)
        self.addCleanup(self.context.stop)