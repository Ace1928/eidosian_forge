import os
import re
import sys
from importlib import util as importlib_util
import breezy
from . import debug, errors, osutils, trace
class _PluginsAtFinder:
    """Meta path finder to support BRZ_PLUGINS_AT configuration."""

    def __init__(self, prefix, names_and_paths):
        self.prefix = prefix
        self.names_to_path = {prefix + n: p for n, p in names_and_paths}

    def __repr__(self):
        return '<{} {!r}>'.format(self.__class__.__name__, self.prefix)

    def find_spec(self, fullname, paths, target=None):
        """New module spec returning find method."""
        if fullname not in self.names_to_path:
            return None
        path = self.names_to_path[fullname]
        if os.path.isdir(path):
            path = _get_package_init(path)
            if path is None:
                raise ImportError('Not loading namespace package {} as {}'.format(path, fullname))
        return importlib_util.spec_from_file_location(fullname, path)