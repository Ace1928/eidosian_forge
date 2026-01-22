from __future__ import annotations
import abc
import argparse
import ast
import datetime
import json
import os
import re
import sys
import traceback
import warnings
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import contextmanager
from fnmatch import fnmatch
from antsibull_docs_parser import dom
from antsibull_docs_parser.parser import parse, Context
import yaml
from voluptuous.humanize import humanize_error
from ansible import __version__ as ansible_version
from ansible.executor.module_common import REPLACER_WINDOWS, NEW_STYLE_PYTHON_MODULE_RE
from ansible.module_utils.common.collections import is_iterable
from ansible.module_utils.common.parameters import DEFAULT_TYPE_VALIDATORS
from ansible.module_utils.compat.version import StrictVersion, LooseVersion
from ansible.module_utils.basic import to_bytes
from ansible.module_utils.six import PY3, with_metaclass, string_types
from ansible.plugins.loader import fragment_loader
from ansible.plugins.list import IGNORE as REJECTLIST
from ansible.utils.plugin_docs import add_collection_to_versions_and_dates, add_fragments, get_docstring
from ansible.utils.version import SemanticVersion
from .module_args import AnsibleModuleImportError, AnsibleModuleNotInitialized, get_argument_spec
from .schema import (
from .utils import CaptureStd, NoArgsAnsibleModule, compare_unordered_lists, parse_yaml, parse_isodate
def _validate_ansible_module_call(self, docs):
    try:
        spec, kwargs = get_argument_spec(self.path, self.collection)
    except AnsibleModuleNotInitialized:
        self.reporter.error(path=self.object_path, code='ansible-module-not-initialized', msg='Execution of the module did not result in initialization of AnsibleModule')
        return
    except AnsibleModuleImportError as e:
        self.reporter.error(path=self.object_path, code='import-error', msg="Exception attempting to import module for argument_spec introspection, '%s'" % e)
        self.reporter.trace(path=self.object_path, tracebk=traceback.format_exc())
        return
    schema = ansible_module_kwargs_schema(self.object_name.split('.')[0], for_collection=bool(self.collection))
    self._validate_docs_schema(kwargs, schema, 'AnsibleModule', 'invalid-ansiblemodule-schema')
    self._validate_argument_spec(docs, spec, kwargs)
    if isinstance(docs, Mapping) and isinstance(docs.get('attributes'), Mapping):
        if isinstance(docs['attributes'].get('check_mode'), Mapping):
            support_value = docs['attributes']['check_mode'].get('support')
            if not kwargs.get('supports_check_mode', False):
                if support_value != 'none':
                    self.reporter.error(path=self.object_path, code='attributes-check-mode', msg="The module does not declare support for check mode, but the check_mode attribute's support value is '%s' and not 'none'" % support_value)
            elif support_value not in ('full', 'partial', 'N/A'):
                self.reporter.error(path=self.object_path, code='attributes-check-mode', msg="The module does declare support for check mode, but the check_mode attribute's support value is '%s'" % support_value)
            if support_value in ('partial', 'N/A') and docs['attributes']['check_mode'].get('details') in (None, '', []):
                self.reporter.error(path=self.object_path, code='attributes-check-mode-details', msg='The module declares it does not fully support check mode, but has no details on what exactly that means')