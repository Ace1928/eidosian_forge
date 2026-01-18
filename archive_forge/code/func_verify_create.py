from unittest import mock
from openstack.tests.unit import base
def verify_create(self, test_method, resource_type, base_path=None, *, method_args=None, method_kwargs=None, expected_args=None, expected_kwargs=None, expected_result='result', mock_method='openstack.proxy.Proxy._create'):
    if method_args is None:
        method_args = []
    if method_kwargs is None:
        method_kwargs = {'x': 1, 'y': 2, 'z': 3}
    if expected_args is None:
        expected_args = method_args.copy()
    if expected_kwargs is None:
        expected_kwargs = method_kwargs.copy()
    expected_kwargs['base_path'] = base_path
    self._verify(mock_method, test_method, method_args=method_args, method_kwargs=method_kwargs, expected_args=[resource_type] + expected_args, expected_kwargs=expected_kwargs, expected_result=expected_result)