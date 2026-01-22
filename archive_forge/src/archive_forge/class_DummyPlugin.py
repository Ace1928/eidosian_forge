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
class DummyPlugin:
    __version__ = '0.1.0'
    module = DummyModule()