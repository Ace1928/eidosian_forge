import copy
from unittest import mock
from oslo_messaging._drivers import common as rpc_common
from oslo_utils import reflection
from heat.common import exception
from heat.common import identifier
from heat.rpc import client as rpc_client
from heat.tests import common
from heat.tests import utils
def test_show_stack(self):
    self._test_engine_api('show_stack', 'call', stack_identity='wordpress', resolve_outputs=True)