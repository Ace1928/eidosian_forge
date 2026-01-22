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
class CoreGitCache(GitCache):
    """Provides access to original files when testing core."""

    def __init__(self, original_plugins: str | None, plugin_type: str) -> None:
        super().__init__()
        self.original_plugins = original_plugins
        rel_path = 'lib/ansible/modules/' if plugin_type == 'module' else f'lib/ansible/plugins/{plugin_type}/'
        head_tree = self._find_files(rel_path)
        head_aliased_modules = set()
        for path in head_tree:
            filename = os.path.basename(path)
            if filename.startswith('_') and filename != '__init__.py':
                if os.path.islink(path):
                    head_aliased_modules.add(os.path.basename(os.path.realpath(path)))
        self._head_aliased_modules = head_aliased_modules

    def get_original_path(self, path: str) -> str | None:
        """Return the path to the original version of the specified file, or None if there isn't one."""
        path = os.path.join(self.original_plugins, path)
        if not os.path.exists(path):
            path = None
        return path

    def is_new(self, path: str) -> bool | None:
        """Return True if the content is new, False if it is not and None if the information is not available."""
        if os.path.basename(path).startswith('_'):
            return False
        if os.path.basename(path) in self._head_aliased_modules:
            return False
        return not self.get_original_path(path)

    @staticmethod
    def _find_files(path: str) -> list[str]:
        """Return a list of files found in the specified directory."""
        paths = []
        for dir_path, dir_names, file_names in os.walk(path):
            for file_name in file_names:
                paths.append(os.path.join(dir_path, file_name))
        return sorted(paths)