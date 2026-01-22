import functools
import importlib.util
import pkgutil
import sys
import types
from oslo_log import log as logging
def _module_name(*components):
    """Assemble a fully-qualified module name from its components."""
    return '.'.join(components)