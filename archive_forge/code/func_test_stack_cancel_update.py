import copy
from unittest import mock
from oslo_messaging._drivers import common as rpc_common
from oslo_utils import reflection
from heat.common import exception
from heat.common import identifier
from heat.rpc import client as rpc_client
from heat.tests import common
from heat.tests import utils
def test_stack_cancel_update(self):
    self._test_engine_api('stack_cancel_update', 'call', stack_identity=self.identity, cancel_with_rollback=False, version='1.14')