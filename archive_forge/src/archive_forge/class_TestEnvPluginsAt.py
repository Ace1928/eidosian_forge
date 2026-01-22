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
class TestEnvPluginsAt(tests.TestCase):

    def _get_paths(self, env_value):
        os.environ['BRZ_PLUGINS_AT'] = env_value
        return plugin._env_plugins_at()

    def test_empty(self):
        self.assertEqual([], plugin._env_plugins_at())
        self.assertEqual([], self._get_paths(''))

    def test_one_path(self):
        self.assertEqual([('b', os.path.abspath('man'))], self._get_paths('b@man'))

    def test_multiple(self):
        self.assertEqual([('tools', os.path.abspath('bzr-tools')), ('p', os.path.abspath('play.py'))], self._get_paths(os.pathsep.join(('tools@bzr-tools', 'p@play.py'))))

    def test_many_at(self):
        self.assertEqual([('church', os.path.abspath('StMichael@Plea@Norwich'))], self._get_paths('church@StMichael@Plea@Norwich'))

    def test_only_py(self):
        self.assertEqual([('test', os.path.abspath('test.py'))], self._get_paths('./test.py'))

    def test_only_package(self):
        self.assertEqual([('py', '/opt/b/py')], self._get_paths('/opt/b/py'))

    def test_bad_name(self):
        self.assertEqual([], self._get_paths('/usr/local/bzr-git'))
        self.assertContainsRe(self.get_log(), "Invalid name 'bzr-git' in BRZ_PLUGINS_AT='/usr/local/bzr-git'")