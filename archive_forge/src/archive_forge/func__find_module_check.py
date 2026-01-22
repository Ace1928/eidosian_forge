import sys
import warnings
import importlib
from contextlib import contextmanager
import gi
from ._gi import Repository, RepositoryError
from ._gi import PyGIWarning
from .module import get_introspection_module
from .overrides import load_overrides
def _find_module_check(self, fullname):
    if not fullname.startswith(self.path):
        return False
    path, namespace = fullname.rsplit('.', 1)
    return path == self.path