import inspect
import re
from oslo_config import cfg
from oslo_log import log
from oslo_messaging._drivers import common as rpc_common
import webob
import heat.api.middleware.fault as fault
from heat.common import exception as heat_exc
from heat.common.i18n import _
from heat.tests import common
def test_openstack_exception_without_kwargs(self):
    wrapper = fault.FaultWrapper(None)
    msg = wrapper._error(heat_exc.StackResourceLimitExceeded())
    expected = {'code': 500, 'error': {'message': 'Maximum resources per stack exceeded.', 'traceback': None, 'type': 'StackResourceLimitExceeded'}, 'explanation': 'The server has either erred or is incapable of performing the requested operation.', 'title': 'Internal Server Error'}
    self.assertEqual(expected, msg)