import importlib
import os
from unittest import mock
from osc_lib.tests import utils as osc_lib_test_utils
import wrapt
from openstackclient import shell
class TestShellTokenAuthEnv(TestShell):

    def setUp(self):
        super(TestShellTokenAuthEnv, self).setUp()
        env = {'OS_TOKEN': DEFAULT_TOKEN, 'OS_AUTH_URL': DEFAULT_AUTH_URL}
        self.useFixture(osc_lib_test_utils.EnvFixture(env.copy()))

    def test_env(self):
        flag = ''
        kwargs = {'token': DEFAULT_TOKEN, 'auth_url': DEFAULT_AUTH_URL}
        self._assert_token_auth(flag, kwargs)

    def test_only_token(self):
        flag = '--os-token xyzpdq'
        kwargs = {'token': 'xyzpdq', 'auth_url': DEFAULT_AUTH_URL}
        self._assert_token_auth(flag, kwargs)

    def test_only_auth_url(self):
        flag = '--os-auth-url http://cloud.local:555'
        kwargs = {'token': DEFAULT_TOKEN, 'auth_url': 'http://cloud.local:555'}
        self._assert_token_auth(flag, kwargs)

    def test_empty_auth(self):
        os.environ = {}
        flag = ''
        kwargs = {'token': '', 'auth_url': ''}
        self._assert_token_auth(flag, kwargs)