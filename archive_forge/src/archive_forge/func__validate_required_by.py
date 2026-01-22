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
def _validate_required_by(self, terms, spec, context):
    if terms is None:
        return
    if not isinstance(terms, Mapping):
        return
    for key, value in terms.items():
        if isinstance(value, string_types):
            value = [value]
        if not isinstance(value, (list, tuple)):
            continue
        for term in value:
            if not isinstance(term, string_types):
                continue
        if len(set(value)) != len(value) or key in value:
            msg = 'required_by'
            if context:
                msg += ' found in %s' % ' -> '.join(context)
            msg += ' has repeated terms'
            self.reporter.error(path=self.object_path, code='required_by-collision', msg=msg)
        if not set(value) <= set(spec) or key not in spec:
            msg = 'required_by'
            if context:
                msg += ' found in %s' % ' -> '.join(context)
            msg += ' contains terms which are not part of argument_spec: %s' % ', '.join(sorted(set(value).difference(set(spec))))
            self.reporter.error(path=self.object_path, code='required_by-unknown', msg=msg)