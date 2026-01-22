from unittest import mock
import uuid
import stevedore
from keystone.api._shared import authentication
from keystone import auth
from keystone.auth.plugins import base
from keystone.auth.plugins import mapped
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit.ksfixtures import auth_plugins
class TestAuthPluginDynamicOptions(TestAuthPlugin):

    def config_overrides(self):
        super(TestAuthPluginDynamicOptions, self).config_overrides()
        self.config_fixture.conf.clear_override('methods', group='auth')

    def config_files(self):
        config_files = super(TestAuthPluginDynamicOptions, self).config_files()
        config_files.append(unit.dirs.tests_conf('test_auth_plugin.conf'))
        return config_files