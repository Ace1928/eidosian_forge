from __future__ import annotations
import abc
import glob
import hashlib
import json
import os
import pathlib
import re
import collections
import collections.abc as c
import typing as t
from ...constants import (
from ...encoding import (
from ...io import (
from ...util import (
from ...util_common import (
from ...ansible_util import (
from ...target import (
from ...executor import (
from ...python_requirements import (
from ...config import (
from ...test import (
from ...data import (
from ...content_config import (
from ...host_configs import (
from ...host_profiles import (
from ...provisioning import (
from ...pypi_proxy import (
from ...venv import (
class SanityIgnoreParser:
    """Parser for the consolidated sanity test ignore file."""
    NO_CODE = '_'

    def __init__(self, args: SanityConfig) -> None:
        if data_context().content.collection:
            ansible_version = '%s.%s' % tuple(get_ansible_version().split('.')[:2])
            ansible_label = 'Ansible %s' % ansible_version
            file_name = 'ignore-%s.txt' % ansible_version
        else:
            ansible_label = 'Ansible'
            file_name = 'ignore.txt'
        self.args = args
        self.relative_path = os.path.join(data_context().content.sanity_path, file_name)
        self.path = os.path.join(data_context().content.root, self.relative_path)
        self.ignores: dict[str, dict[str, dict[str, int]]] = collections.defaultdict(lambda: collections.defaultdict(dict))
        self.skips: dict[str, dict[str, int]] = collections.defaultdict(lambda: collections.defaultdict(int))
        self.parse_errors: list[tuple[int, int, str]] = []
        self.file_not_found_errors: list[tuple[int, str]] = []
        lines = read_lines_without_comments(self.path, optional=True)
        targets = SanityTargets.get_targets()
        paths = set((target.path for target in targets))
        tests_by_name: dict[str, SanityTest] = {}
        versioned_test_names: set[str] = set()
        unversioned_test_names: dict[str, str] = {}
        directories = paths_to_dirs(list(paths))
        paths_by_test: dict[str, set[str]] = {}
        display.info('Read %d sanity test ignore line(s) for %s from: %s' % (len(lines), ansible_label, self.relative_path), verbosity=1)
        for test in sanity_get_tests():
            test_targets = SanityTargets.filter_and_inject_targets(test, targets)
            if isinstance(test, SanityMultipleVersion):
                versioned_test_names.add(test.name)
                for python_version in test.supported_python_versions:
                    test_name = '%s-%s' % (test.name, python_version)
                    paths_by_test[test_name] = set((target.path for target in test.filter_targets_by_version(args, test_targets, python_version)))
                    tests_by_name[test_name] = test
            else:
                unversioned_test_names.update(dict((('%s-%s' % (test.name, python_version), test.name) for python_version in SUPPORTED_PYTHON_VERSIONS)))
                paths_by_test[test.name] = set((target.path for target in test.filter_targets_by_version(args, test_targets, '')))
                tests_by_name[test.name] = test
        for line_no, line in enumerate(lines, start=1):
            if not line:
                self.parse_errors.append((line_no, 1, 'Line cannot be empty or contain only a comment'))
                continue
            parts = line.split(' ')
            path = parts[0]
            codes = parts[1:]
            if not path:
                self.parse_errors.append((line_no, 1, 'Line cannot start with a space'))
                continue
            if path.endswith(os.path.sep):
                if path not in directories:
                    self.file_not_found_errors.append((line_no, path))
                    continue
            elif path not in paths:
                self.file_not_found_errors.append((line_no, path))
                continue
            if not codes:
                self.parse_errors.append((line_no, len(path), 'Error code required after path'))
                continue
            code = codes[0]
            if not code:
                self.parse_errors.append((line_no, len(path) + 1, 'Error code after path cannot be empty'))
                continue
            if len(codes) > 1:
                self.parse_errors.append((line_no, len(path) + len(code) + 2, 'Error code cannot contain spaces'))
                continue
            parts = code.split('!')
            code = parts[0]
            commands = parts[1:]
            parts = code.split(':')
            test_name = parts[0]
            error_codes = parts[1:]
            test = tests_by_name.get(test_name)
            if not test:
                unversioned_name = unversioned_test_names.get(test_name)
                if unversioned_name:
                    self.parse_errors.append((line_no, len(path) + len(unversioned_name) + 2, "Sanity test '%s' cannot use a Python version like '%s'" % (unversioned_name, test_name)))
                elif test_name in versioned_test_names:
                    self.parse_errors.append((line_no, len(path) + len(test_name) + 1, "Sanity test '%s' requires a Python version like '%s-%s'" % (test_name, test_name, args.controller_python.version)))
                else:
                    self.parse_errors.append((line_no, len(path) + 2, "Sanity test '%s' does not exist" % test_name))
                continue
            if path.endswith(os.path.sep) and (not test.include_directories):
                self.parse_errors.append((line_no, 1, "Sanity test '%s' does not support directory paths" % test_name))
                continue
            if path not in paths_by_test[test_name] and (not test.no_targets):
                self.parse_errors.append((line_no, 1, "Sanity test '%s' does not test path '%s'" % (test_name, path)))
                continue
            if commands and error_codes:
                self.parse_errors.append((line_no, len(path) + len(test_name) + 2, "Error code cannot contain both '!' and ':' characters"))
                continue
            if commands:
                command = commands[0]
                if len(commands) > 1:
                    self.parse_errors.append((line_no, len(path) + len(test_name) + len(command) + 3, "Error code cannot contain multiple '!' characters"))
                    continue
                if command == 'skip':
                    if not test.can_skip:
                        self.parse_errors.append((line_no, len(path) + len(test_name) + 2, "Sanity test '%s' cannot be skipped" % test_name))
                        continue
                    existing_line_no = self.skips.get(test_name, {}).get(path)
                    if existing_line_no:
                        self.parse_errors.append((line_no, 1, "Duplicate '%s' skip for path '%s' first found on line %d" % (test_name, path, existing_line_no)))
                        continue
                    self.skips[test_name][path] = line_no
                    continue
                self.parse_errors.append((line_no, len(path) + len(test_name) + 2, "Command '!%s' not recognized" % command))
                continue
            if not test.can_ignore:
                self.parse_errors.append((line_no, len(path) + 1, "Sanity test '%s' cannot be ignored" % test_name))
                continue
            if test.error_code:
                if not error_codes:
                    self.parse_errors.append((line_no, len(path) + len(test_name) + 1, "Sanity test '%s' requires an error code" % test_name))
                    continue
                error_code = error_codes[0]
                if len(error_codes) > 1:
                    self.parse_errors.append((line_no, len(path) + len(test_name) + len(error_code) + 3, "Error code cannot contain multiple ':' characters"))
                    continue
                if error_code in test.optional_error_codes:
                    self.parse_errors.append((line_no, len(path) + len(test_name) + 3, "Optional error code '%s' cannot be ignored" % error_code))
                    continue
            else:
                if error_codes:
                    self.parse_errors.append((line_no, len(path) + len(test_name) + 2, "Sanity test '%s' does not support error codes" % test_name))
                    continue
                error_code = self.NO_CODE
            existing = self.ignores.get(test_name, {}).get(path, {}).get(error_code)
            if existing:
                if test.error_code:
                    self.parse_errors.append((line_no, 1, "Duplicate '%s' ignore for error code '%s' for path '%s' first found on line %d" % (test_name, error_code, path, existing)))
                else:
                    self.parse_errors.append((line_no, 1, "Duplicate '%s' ignore for path '%s' first found on line %d" % (test_name, path, existing)))
                continue
            self.ignores[test_name][path][error_code] = line_no

    @staticmethod
    def load(args: SanityConfig) -> SanityIgnoreParser:
        """Return the current SanityIgnore instance, initializing it if needed."""
        try:
            return SanityIgnoreParser.instance
        except AttributeError:
            pass
        instance = SanityIgnoreParser(args)
        SanityIgnoreParser.instance = instance
        return instance