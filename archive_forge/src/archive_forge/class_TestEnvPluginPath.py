import importlib
import logging
import os
import sys
import types
from io import StringIO
from typing import Any, Dict, List
import breezy
from .. import osutils, plugin, tests
from . import test_bar
class TestEnvPluginPath(tests.TestCase):
    user = 'USER'
    core = 'CORE'
    site = 'SITE'

    def check_path(self, expected_dirs, setting_dirs):
        if setting_dirs is None:
            del os.environ['BRZ_PLUGIN_PATH']
        else:
            os.environ['BRZ_PLUGIN_PATH'] = os.pathsep.join(setting_dirs)
        actual = [p if t == 'path' else t.upper() for p, t in plugin._env_plugin_path()]
        self.assertEqual(expected_dirs, actual)

    def test_default(self):
        self.check_path([self.user, self.core, self.site], None)

    def test_adhoc_policy(self):
        self.check_path([self.user, self.core, self.site], ['+user', '+core', '+site'])

    def test_fallback_policy(self):
        self.check_path([self.core, self.site, self.user], ['+core', '+site', '+user'])

    def test_override_policy(self):
        self.check_path([self.user, self.site, self.core], ['+user', '+site', '+core'])

    def test_disable_user(self):
        self.check_path([self.core, self.site], ['-user'])

    def test_disable_user_twice(self):
        self.check_path([self.core, self.site], ['-user', '-user'])

    def test_duplicates_are_removed(self):
        self.check_path([self.user, self.core, self.site], ['+user', '+user'])
        self.check_path([self.user, self.core, self.site], ['+user', '+user', '+core', '+user', '+site', '+site', '+core'])

    def test_disable_overrides_enable(self):
        self.check_path([self.core, self.site], ['-user', '+user'])

    def test_disable_core(self):
        self.check_path([self.site], ['-core'])
        self.check_path([self.user, self.site], ['+user', '-core'])

    def test_disable_site(self):
        self.check_path([self.core], ['-site'])
        self.check_path([self.user, self.core], ['-site', '+user'])

    def test_override_site(self):
        self.check_path(['mysite', self.user, self.core], ['mysite', '-site', '+user'])
        self.check_path(['mysite', self.core], ['mysite', '-site'])

    def test_override_core(self):
        self.check_path(['mycore', self.user, self.site], ['mycore', '-core', '+user', '+site'])
        self.check_path(['mycore', self.site], ['mycore', '-core'])

    def test_my_plugin_only(self):
        self.check_path(['myplugin'], ['myplugin', '-user', '-core', '-site'])

    def test_my_plugin_first(self):
        self.check_path(['myplugin', self.core, self.site, self.user], ['myplugin', '+core', '+site', '+user'])

    def test_bogus_references(self):
        self.check_path(['+foo', '-bar', self.core, self.site], ['+foo', '-bar'])