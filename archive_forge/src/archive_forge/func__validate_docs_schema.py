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
def _validate_docs_schema(self, doc, schema, name, error_code):
    errors = []
    try:
        schema(doc)
    except Exception as e:
        for error in e.errors:
            error.data = doc
        errors.extend(e.errors)
    for error in errors:
        path = [str(p) for p in error.path]
        local_error_code = getattr(error, 'ansible_error_code', error_code)
        if isinstance(error.data, dict):
            error_message = humanize_error(error.data, error)
        else:
            error_message = error
        if path:
            combined_path = '%s.%s' % (name, '.'.join(path))
        else:
            combined_path = name
        self.reporter.error(path=self.object_path, code=local_error_code, msg='%s: %s' % (combined_path, error_message))