from unittest import mock
from oslo_config import cfg
from oslo_context import context as oslo_context
from testtools import matchers
from neutron_lib import context
from neutron_lib.tests import _base
def test_session_cached(self):
    ctx = context.Context('user_id', 'tenant_id')
    session1 = ctx.session
    session2 = ctx.session
    self.assertIs(session1, session2)