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
def test_short_version_info_gets_padded(self):
    self.check_version_info((1, 2, 3, 'final', 0), 'version_info = (1, 2, 3)')