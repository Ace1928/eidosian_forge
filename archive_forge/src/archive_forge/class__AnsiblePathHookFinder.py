from __future__ import (absolute_import, division, print_function)
import itertools
import os
import os.path
import pkgutil
import re
import sys
from keyword import iskeyword
from tokenize import Name as _VALID_IDENTIFIER_REGEX
from ansible.module_utils.common.text.converters import to_native, to_text, to_bytes
from ansible.module_utils.six import string_types, PY3
from ._collection_config import AnsibleCollectionConfig
from contextlib import contextmanager
from types import ModuleType
class _AnsiblePathHookFinder:

    def __init__(self, collection_finder, pathctx):
        self._pathctx = to_native(pathctx)
        self._collection_finder = collection_finder
        if PY3:
            self._file_finder = None

    def _get_filefinder_path_hook(self=None):
        _file_finder_hook = None
        if PY3:
            _file_finder_hook = [ph for ph in sys.path_hooks if 'FileFinder' in repr(ph)]
            if len(_file_finder_hook) != 1:
                raise Exception('need exactly one FileFinder import hook (found {0})'.format(len(_file_finder_hook)))
            _file_finder_hook = _file_finder_hook[0]
        return _file_finder_hook
    _filefinder_path_hook = _get_filefinder_path_hook()

    def _get_finder(self, fullname):
        split_name = fullname.split('.')
        toplevel_pkg = split_name[0]
        if toplevel_pkg == 'ansible_collections':
            return self._collection_finder
        else:
            if PY3:
                if not self._file_finder:
                    try:
                        self._file_finder = _AnsiblePathHookFinder._filefinder_path_hook(self._pathctx)
                    except ImportError:
                        return None
                return self._file_finder
            return pkgutil.ImpImporter(self._pathctx)

    def find_module(self, fullname, path=None):
        finder = self._get_finder(fullname)
        if finder is None:
            return None
        elif HAS_FILE_FINDER and isinstance(finder, FileFinder):
            return finder.find_module(fullname)
        else:
            return finder.find_module(fullname, path=[self._pathctx])

    def find_spec(self, fullname, target=None):
        split_name = fullname.split('.')
        toplevel_pkg = split_name[0]
        finder = self._get_finder(fullname)
        if finder is None:
            return None
        elif toplevel_pkg == 'ansible_collections':
            return finder.find_spec(fullname, path=[self._pathctx])
        else:
            return finder.find_spec(fullname)

    def iter_modules(self, prefix):
        return _iter_modules_impl([self._pathctx], prefix)

    def __repr__(self):
        return "{0}(path='{1}')".format(self.__class__.__name__, self._pathctx)