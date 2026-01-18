from unittest import mock
from openstack.tests.unit import base
def verify_head(self, test_method, resource_type, base_path=None, *, method_args=None, method_kwargs=None, expected_args=None, expected_kwargs=None, mock_method='openstack.proxy.Proxy._head'):
    if method_args is None:
        method_args = ['resource_id']
    if method_kwargs is None:
        method_kwargs = {}
    expected_args = expected_args or method_args.copy()
    expected_kwargs = expected_kwargs or method_kwargs.copy()
    self._verify(mock_method, test_method, method_args=method_args, method_kwargs=method_kwargs, expected_args=[resource_type] + expected_args, expected_kwargs=expected_kwargs)