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
def test_plugin_with_error(self):
    name = 'some_error.py'
    with open(name, 'w') as f:
        f.write('raise Exception("bad")\n')
    log = self.load_and_capture(name, warn_load_problems=True)
    self.assertContainsRe(log, "Unable to load plugin 'some_error' from '.*': bad\n")