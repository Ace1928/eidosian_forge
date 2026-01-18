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
def test_regular_load(self):
    self.load_with_paths(['standard'])
    self.assertTestFooLoadedFrom('standard/test_foo')