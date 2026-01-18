from unittest import mock
from openstack.tests.unit import base
def verify_get(self, test_method, resource_type, requires_id=False, base_path=None, *, method_args=None, method_kwargs=None, expected_args=None, expected_kwargs=None, mock_method='openstack.proxy.Proxy._get'):
    if method_args is None:
        method_args = ['resource_id']
    if method_kwargs is None:
        method_kwargs = {}
    if expected_args is None:
        expected_args = method_args.copy()
    if expected_kwargs is None:
        expected_kwargs = method_kwargs.copy()
    self._verify(mock_method, test_method, method_args=method_args, method_kwargs=method_kwargs, expected_args=[resource_type] + expected_args, expected_kwargs=expected_kwargs)