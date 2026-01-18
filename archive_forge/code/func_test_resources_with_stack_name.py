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
@mock.patch.object(stack.Stack, 'find')
def test_resources_with_stack_name(self, mock_find):
    stack_id = '1234'
    stack_name = 'test_stack'
    stk = stack.Stack(id=stack_id, name=stack_name)
    mock_find.return_value = stk
    self.verify_list(self.proxy.resources, resource.Resource, method_args=[stack_id], expected_args=[], expected_kwargs={'stack_name': stack_name, 'stack_id': stack_id})
    mock_find.assert_called_once_with(mock.ANY, stack_id, ignore_missing=False)