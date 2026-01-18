import importlib
import os
from unittest import mock
from osc_lib.tests import utils as osc_lib_test_utils
import wrapt
from openstackclient import shell
def test_only_token(self):
    flag = '--os-token xyzpdq'
    kwargs = {'token': 'xyzpdq', 'endpoint': DEFAULT_SERVICE_URL}
    self._assert_token_auth(flag, kwargs)