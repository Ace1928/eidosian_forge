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
def test_plugin_with_bad_api_version_reports(self):
    """Try loading a plugin that requests an unsupported api.

        Observe that it records the problem but doesn't complain on stderr
        when warn_load_problems=False
        """
    name = 'wants100.py'
    with open(name, 'w') as f:
        f.write('import breezy\nfrom breezy.errors import IncompatibleVersion\nraise IncompatibleVersion(breezy, [(1, 0, 0)], (0, 0, 5))\n')
    log = self.load_and_capture(name, warn_load_problems=False)
    self.assertNotContainsRe(log, 'It supports breezy version')
    self.assertEqual({'wants100'}, self.plugin_warnings.keys())
    self.assertContainsRe(self.plugin_warnings['wants100'][0], 'It supports breezy version')