from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import importlib
import importlib.util
import os
import sys
from googlecloudsdk.core import exceptions
import six
def _GetPrivateModulePath(module_path):
    """Mock hook that returns the module path for module that starts with '__'."""
    del module_path
    return None