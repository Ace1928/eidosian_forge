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
def test_should_not_ignore_parent_classes_even_for_remote_ones(self):
    cfg.CONF.set_override('debug', True)
    error = StackNotFoundChild(entity='Stack', name='a')
    exc_info = (type(error), error, None)
    serialized = rpc_common.serialize_remote_exception(exc_info)
    remote_error = rpc_common.deserialize_remote_exception(serialized, ['heat.tests.test_fault_middleware'])
    wrapper = fault.FaultWrapper(None)
    msg = wrapper._error(remote_error)
    expected_message, expected_traceback = str(remote_error).split('\n', 1)
    expected = {'code': 404, 'error': {'message': expected_message, 'traceback': expected_traceback, 'type': 'StackNotFoundChild'}, 'explanation': 'The resource could not be found.', 'title': 'Not Found'}
    self.assertEqual(expected, msg)