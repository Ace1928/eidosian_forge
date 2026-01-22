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
class TestEnvDisablePlugins(tests.TestCase):

    def _get_names(self, env_value):
        os.environ['BRZ_DISABLE_PLUGINS'] = env_value
        return plugin._env_disable_plugins()

    def test_unset(self):
        self.assertEqual([], plugin._env_disable_plugins())

    def test_empty(self):
        self.assertEqual([], self._get_names(''))

    def test_single(self):
        self.assertEqual(['single'], self._get_names('single'))

    def test_multi(self):
        expected = ['one', 'two']
        self.assertEqual(expected, self._get_names(os.pathsep.join(expected)))

    def test_mixed(self):
        value = os.pathsep.join(['valid', 'in-valid'])
        self.assertEqual(['valid'], self._get_names(value))
        self.assertContainsRe(self.get_log(), "Invalid name 'in-valid' in BRZ_DISABLE_PLUGINS=" + repr(value))