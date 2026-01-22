from __future__ import annotations
import os
import typing as t
import astroid
from pylint.checkers import BaseChecker
class UnwantedEntry:
    """Defines an unwanted import."""

    def __init__(self, alternative, modules_only=False, names=None, ignore_paths=None, ansible_test_only=False):
        self.alternative = alternative
        self.modules_only = modules_only
        self.names = set(names) if names else set()
        self.ignore_paths = ignore_paths
        self.ansible_test_only = ansible_test_only

    def applies_to(self, path, name=None):
        """Return True if this entry applies to the given path, otherwise return False."""
        if self.names:
            if not name:
                return False
            if name not in self.names:
                return False
        if self.ignore_paths and any((path.endswith(ignore_path) for ignore_path in self.ignore_paths)):
            return False
        if self.ansible_test_only and '/test/lib/ansible_test/_internal/' not in path:
            return False
        if self.modules_only:
            return is_module_path(path)
        return True