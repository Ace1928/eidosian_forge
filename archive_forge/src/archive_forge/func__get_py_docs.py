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
def _get_py_docs(self):
    docs = {'DOCUMENTATION': {'value': None, 'lineno': 0, 'end_lineno': 0}, 'EXAMPLES': {'value': None, 'lineno': 0, 'end_lineno': 0}, 'RETURN': {'value': None, 'lineno': 0, 'end_lineno': 0}}
    for child in self.ast.body:
        if isinstance(child, ast.Assign):
            for grandchild in child.targets:
                if not isinstance(grandchild, ast.Name):
                    continue
                if grandchild.id == 'DOCUMENTATION':
                    docs['DOCUMENTATION']['value'] = child.value.value
                    docs['DOCUMENTATION']['lineno'] = child.lineno
                    docs['DOCUMENTATION']['end_lineno'] = child.lineno + len(child.value.value.splitlines())
                elif grandchild.id == 'EXAMPLES':
                    docs['EXAMPLES']['value'] = child.value.value
                    docs['EXAMPLES']['lineno'] = child.lineno
                    docs['EXAMPLES']['end_lineno'] = child.lineno + len(child.value.value.splitlines())
                elif grandchild.id == 'RETURN':
                    docs['RETURN']['value'] = child.value.value
                    docs['RETURN']['lineno'] = child.lineno
                    docs['RETURN']['end_lineno'] = child.lineno + len(child.value.value.splitlines())
    return docs