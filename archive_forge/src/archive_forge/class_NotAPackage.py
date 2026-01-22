import os
import sys
import warnings
from contextlib import contextmanager
from importlib import import_module, reload
from kombu.utils.imports import symbol_by_name
class NotAPackage(Exception):
    """Raised when importing a package, but it's not a package."""