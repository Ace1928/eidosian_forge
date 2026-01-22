import os
from unittest import mock
import fixtures
from oslo_serialization import jsonutils
import requests
from requests_mock.contrib import fixture as requests_mock_fixture
import testscenarios
import testtools
def _patch_mock_to_raise_for_invalid_assert_calls():

    def raise_for_invalid_assert_calls(wrapped):

        def wrapper(_self, name):
            valid_asserts = ['assert_called_with', 'assert_called_once_with', 'assert_has_calls', 'assert_any_calls']
            if name.startswith('assert') and name not in valid_asserts:
                raise AttributeError('%s is not a valid mock assert method' % name)
            return wrapped(_self, name)
        return wrapper
    mock.Mock.__getattr__ = raise_for_invalid_assert_calls(mock.Mock.__getattr__)