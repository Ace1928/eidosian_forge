from __future__ import (absolute_import, division, print_function)
import ast
import base64
import datetime
import json
import os
import shlex
import time
import zipfile
import re
import pkgutil
from ast import AST, Import, ImportFrom
from io import BytesIO
from ansible.release import __version__, __author__
from ansible import constants as C
from ansible.errors import AnsibleError
from ansible.executor.interpreter_discovery import InterpreterDiscoveryRequiredError
from ansible.executor.powershell import module_manifest as ps_manifest
from ansible.module_utils.common.json import AnsibleJSONEncoder
from ansible.module_utils.common.text.converters import to_bytes, to_text, to_native
from ansible.plugins.loader import module_utils_loader
from ansible.utils.collection_loader._collection_finder import _get_collection_metadata, _nested_dict_get
from ansible.executor import action_write_locks
from ansible.utils.display import Display
from collections import namedtuple
import importlib.util
import importlib.machinery
import sys
import {1} as mod
def _handle_redirect(self, name_parts):
    module_utils_relative_parts = self._get_module_utils_remainder_parts(name_parts)
    if not module_utils_relative_parts:
        return False
    try:
        collection_metadata = _get_collection_metadata(self._collection_name)
    except ValueError as ve:
        if self._is_optional:
            return False
        raise AnsibleError('error processing module_util {0} loading redirected collection {1}: {2}'.format('.'.join(name_parts), self._collection_name, to_native(ve)))
    routing_entry = _nested_dict_get(collection_metadata, ['plugin_routing', 'module_utils', '.'.join(module_utils_relative_parts)])
    if not routing_entry:
        return False
    dep_or_ts = routing_entry.get('tombstone')
    removed = dep_or_ts is not None
    if not removed:
        dep_or_ts = routing_entry.get('deprecation')
    if dep_or_ts:
        removal_date = dep_or_ts.get('removal_date')
        removal_version = dep_or_ts.get('removal_version')
        warning_text = dep_or_ts.get('warning_text')
        msg = 'module_util {0} has been removed'.format('.'.join(name_parts))
        if warning_text:
            msg += ' ({0})'.format(warning_text)
        else:
            msg += '.'
        display.deprecated(msg, removal_version, removed, removal_date, self._collection_name)
    if 'redirect' in routing_entry:
        self.redirected = True
        source_pkg = '.'.join(name_parts)
        self.is_package = True
        redirect_target_pkg = routing_entry['redirect']
        if not redirect_target_pkg.startswith('ansible_collections'):
            split_fqcn = redirect_target_pkg.split('.')
            if len(split_fqcn) < 3:
                raise Exception('invalid redirect for {0}: {1}'.format(source_pkg, redirect_target_pkg))
            redirect_target_pkg = 'ansible_collections.{0}.{1}.plugins.module_utils.{2}'.format(split_fqcn[0], split_fqcn[1], '.'.join(split_fqcn[2:]))
        display.vvv('redirecting module_util {0} to {1}'.format(source_pkg, redirect_target_pkg))
        self.source_code = self._generate_redirect_shim_source(source_pkg, redirect_target_pkg)
        return True
    return False