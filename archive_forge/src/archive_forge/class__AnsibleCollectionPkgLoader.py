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
class _AnsibleCollectionPkgLoader(_AnsibleCollectionPkgLoaderBase):

    def _validate_args(self):
        super(_AnsibleCollectionPkgLoader, self)._validate_args()
        if len(self._split_name) != 3:
            raise ImportError('this loader can only load collection packages, not {0}'.format(self._fullname))

    def _validate_final(self):
        if self._split_name[1:3] == ['ansible', 'builtin']:
            self._subpackage_search_paths = []
        elif not self._subpackage_search_paths:
            raise ImportError('no {0} found in {1}'.format(self._package_to_load, self._candidate_paths))
        else:
            self._subpackage_search_paths = [self._subpackage_search_paths[0]]

    def _load_module(self, module):
        if not _meta_yml_to_dict:
            raise ValueError('ansible.utils.collection_loader._meta_yml_to_dict is not set')
        module._collection_meta = {}
        collection_name = '.'.join(self._split_name[1:3])
        if collection_name == 'ansible.builtin':
            ansible_pkg_path = os.path.dirname(import_module('ansible').__file__)
            metadata_path = os.path.join(ansible_pkg_path, 'config/ansible_builtin_runtime.yml')
            with open(to_bytes(metadata_path), 'rb') as fd:
                raw_routing = fd.read()
        else:
            b_routing_meta_path = to_bytes(os.path.join(module.__path__[0], 'meta/runtime.yml'))
            if os.path.isfile(b_routing_meta_path):
                with open(b_routing_meta_path, 'rb') as fd:
                    raw_routing = fd.read()
            else:
                raw_routing = ''
        try:
            if raw_routing:
                routing_dict = _meta_yml_to_dict(raw_routing, (collection_name, 'runtime.yml'))
                module._collection_meta = self._canonicalize_meta(routing_dict)
        except Exception as ex:
            raise ValueError('error parsing collection metadata: {0}'.format(to_native(ex)))
        AnsibleCollectionConfig.on_collection_load.fire(collection_name=collection_name, collection_path=os.path.dirname(module.__file__))
        return module

    def exec_module(self, module):
        super(_AnsibleCollectionPkgLoader, self).exec_module(module)
        self._load_module(module)

    def create_module(self, spec):
        return None

    def load_module(self, fullname):
        module = super(_AnsibleCollectionPkgLoader, self).load_module(fullname)
        return self._load_module(module)

    def _canonicalize_meta(self, meta_dict):
        return meta_dict