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
class _AnsibleCollectionPkgLoaderBase:
    _allows_package_code = False

    def __init__(self, fullname, path_list=None):
        self._fullname = fullname
        self._redirect_module = None
        self._split_name = fullname.split('.')
        self._rpart_name = fullname.rpartition('.')
        self._parent_package_name = self._rpart_name[0]
        self._package_to_load = self._rpart_name[2]
        self._source_code_path = None
        self._decoded_source = None
        self._compiled_code = None
        self._validate_args()
        self._candidate_paths = self._get_candidate_paths([to_native(p) for p in path_list])
        self._subpackage_search_paths = self._get_subpackage_search_paths(self._candidate_paths)
        self._validate_final()

    def _validate_args(self):
        if self._split_name[0] != 'ansible_collections':
            raise ImportError('this loader can only load packages from the ansible_collections package, not {0}'.format(self._fullname))

    def _get_candidate_paths(self, path_list):
        return [os.path.join(p, self._package_to_load) for p in path_list]

    def _get_subpackage_search_paths(self, candidate_paths):
        return [p for p in candidate_paths if os.path.isdir(to_bytes(p))]

    def _validate_final(self):
        return

    @staticmethod
    @contextmanager
    def _new_or_existing_module(name, **kwargs):
        created_module = False
        module = sys.modules.get(name)
        try:
            if not module:
                module = ModuleType(name)
                created_module = True
                sys.modules[name] = module
            for attr, value in kwargs.items():
                setattr(module, attr, value)
            yield module
        except Exception:
            if created_module:
                if sys.modules.get(name):
                    sys.modules.pop(name)
            raise

    @staticmethod
    def _module_file_from_path(leaf_name, path):
        has_code = True
        package_path = os.path.join(to_native(path), to_native(leaf_name))
        module_path = None
        if os.path.isdir(to_bytes(package_path)):
            module_path = os.path.join(package_path, '__init__.py')
            if not os.path.isfile(to_bytes(module_path)):
                module_path = os.path.join(package_path, '__synthetic__')
                has_code = False
        else:
            module_path = package_path + '.py'
            package_path = None
            if not os.path.isfile(to_bytes(module_path)):
                raise ImportError('{0} not found at {1}'.format(leaf_name, path))
        return (module_path, has_code, package_path)

    def get_resource_reader(self, fullname):
        return _AnsibleTraversableResources(fullname, self)

    def exec_module(self, module):
        if self._redirect_module:
            return
        code_obj = self.get_code(self._fullname)
        if code_obj is not None:
            exec(code_obj, module.__dict__)

    def create_module(self, spec):
        if self._redirect_module:
            return self._redirect_module
        else:
            return None

    def load_module(self, fullname):
        if self._redirect_module:
            sys.modules[self._fullname] = self._redirect_module
            return self._redirect_module
        module_attrs = dict(__loader__=self, __file__=self.get_filename(fullname), __package__=self._parent_package_name)
        if self._subpackage_search_paths is not None:
            module_attrs['__path__'] = self._subpackage_search_paths
            module_attrs['__package__'] = fullname
        with self._new_or_existing_module(fullname, **module_attrs) as module:
            code_obj = self.get_code(fullname)
            if code_obj is not None:
                exec(code_obj, module.__dict__)
            return module

    def is_package(self, fullname):
        if fullname != self._fullname:
            raise ValueError('this loader cannot answer is_package for {0}, only {1}'.format(fullname, self._fullname))
        return self._subpackage_search_paths is not None

    def get_source(self, fullname):
        if self._decoded_source:
            return self._decoded_source
        if fullname != self._fullname:
            raise ValueError('this loader cannot load source for {0}, only {1}'.format(fullname, self._fullname))
        if not self._source_code_path:
            return None
        self._decoded_source = self.get_data(self._source_code_path)
        return self._decoded_source

    def get_data(self, path):
        if not path:
            raise ValueError('a path must be specified')
        if not path[0] == '/':
            raise ValueError('relative resource paths not supported')
        else:
            candidate_paths = [path]
        for p in candidate_paths:
            b_path = to_bytes(p)
            if os.path.isfile(b_path):
                with open(b_path, 'rb') as fd:
                    return fd.read()
            elif b_path.endswith(b'__init__.py') and os.path.isdir(os.path.dirname(b_path)):
                return ''
        return None

    def _synthetic_filename(self, fullname):
        return SYNTHETIC_PACKAGE_NAME

    def get_filename(self, fullname):
        if fullname != self._fullname:
            raise ValueError('this loader cannot find files for {0}, only {1}'.format(fullname, self._fullname))
        filename = self._source_code_path
        if not filename and self.is_package(fullname):
            if len(self._subpackage_search_paths) == 1:
                filename = os.path.join(self._subpackage_search_paths[0], '__synthetic__')
            else:
                filename = self._synthetic_filename(fullname)
        return filename

    def get_code(self, fullname):
        if self._compiled_code:
            return self._compiled_code
        filename = self.get_filename(fullname)
        if not filename:
            filename = '<string>'
        source_code = self.get_source(fullname)
        if source_code is None:
            return None
        self._compiled_code = compile(source=source_code, filename=filename, mode='exec', flags=0, dont_inherit=True)
        return self._compiled_code

    def iter_modules(self, prefix):
        return _iter_modules_impl(self._subpackage_search_paths, prefix)

    def __repr__(self):
        return '{0}(path={1})'.format(self.__class__.__name__, self._subpackage_search_paths or self._source_code_path)