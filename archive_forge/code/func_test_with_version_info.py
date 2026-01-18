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
def test_with_version_info(self):
    self.check_version_info((1, 2, 3, 'dev', 4), "version_info = (1, 2, 3, 'dev', 4)")