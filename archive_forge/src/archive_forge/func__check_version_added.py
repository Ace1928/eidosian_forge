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
def _check_version_added(self, doc, existing_doc):
    version_added_raw = doc.get('version_added')
    try:
        collection_name = doc.get('version_added_collection')
        version_added = self._create_strict_version(str(version_added_raw or '0.0'), collection_name=collection_name)
    except ValueError as e:
        version_added = version_added_raw or '0.0'
        if self._is_new_module() or version_added != 'historical':
            if version_added == 'historical':
                self.reporter.error(path=self.object_path, code='module-invalid-version-added', msg='version_added is not a valid version number: %r. Error: %s' % (version_added, e))
            return
    if existing_doc and str(version_added_raw) != str(existing_doc.get('version_added')):
        self.reporter.error(path=self.object_path, code='module-incorrect-version-added', msg='version_added should be %r. Currently %r' % (existing_doc.get('version_added'), version_added_raw))
    if not self._is_new_module():
        return
    should_be = '.'.join(ansible_version.split('.')[:2])
    strict_ansible_version = self._create_strict_version(should_be, collection_name='ansible.builtin')
    if version_added < strict_ansible_version or strict_ansible_version < version_added:
        self.reporter.error(path=self.object_path, code='module-incorrect-version-added', msg='version_added should be %r. Currently %r' % (should_be, version_added_raw))