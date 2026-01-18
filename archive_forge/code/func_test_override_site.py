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
def test_override_site(self):
    self.check_path(['mysite', self.user, self.core], ['mysite', '-site', '+user'])
    self.check_path(['mysite', self.core], ['mysite', '-site'])