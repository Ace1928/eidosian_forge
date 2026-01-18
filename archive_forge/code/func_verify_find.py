from unittest import mock
from openstack.tests.unit import base
def verify_find(self, test_method, resource_type, name_or_id='resource_name', ignore_missing=True, *, method_args=None, method_kwargs=None, expected_args=None, expected_kwargs=None, mock_method='openstack.proxy.Proxy._find'):
    method_args = [name_or_id] + (method_args or [])
    method_kwargs = method_kwargs or {}
    method_kwargs['ignore_missing'] = ignore_missing
    expected_args = expected_args or method_args.copy()
    expected_kwargs = expected_kwargs or method_kwargs.copy()
    self._verify(mock_method, test_method, method_args=method_args, method_kwargs=method_kwargs, expected_args=[resource_type] + expected_args, expected_kwargs=expected_kwargs)