from unittest import mock
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack import exceptions
from openstack.orchestration.v1 import _proxy
from openstack.orchestration.v1 import resource
from openstack.orchestration.v1 import software_config as sc
from openstack.orchestration.v1 import software_deployment as sd
from openstack.orchestration.v1 import stack
from openstack.orchestration.v1 import stack_environment
from openstack.orchestration.v1 import stack_event
from openstack.orchestration.v1 import stack_files
from openstack.orchestration.v1 import stack_template
from openstack.orchestration.v1 import template
from openstack import proxy
from openstack.tests.unit import test_proxy_base
@mock.patch.object(proxy.Proxy, '_get')
def test_stack_events_with_stack_id(self, mock_get):
    stack_id = '1234'
    stack_name = 'test_stack'
    stk = stack.Stack(id=stack_id, name=stack_name)
    mock_get.return_value = stk
    self._verify('openstack.proxy.Proxy._list', self.proxy.stack_events, method_args=[stk], expected_args=[stack_event.StackEvent], expected_kwargs={'stack_name': stack_name, 'stack_id': stack_id})