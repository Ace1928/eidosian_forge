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
def test_only_py(self):
    self.assertEqual([('test', os.path.abspath('test.py'))], self._get_paths('./test.py'))