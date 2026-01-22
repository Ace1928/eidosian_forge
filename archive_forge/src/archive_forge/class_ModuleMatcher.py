from __future__ import annotations
import hashlib
import ntpath
import os
import os.path
import posixpath
import re
import sys
from typing import Callable, Iterable
from coverage import env
from coverage.exceptions import ConfigError
from coverage.misc import human_sorted, isolate_module, join_regex
class ModuleMatcher:
    """A matcher for modules in a tree."""

    def __init__(self, module_names: Iterable[str], name: str='unknown') -> None:
        self.modules = list(module_names)
        self.name = name

    def __repr__(self) -> str:
        return f'<ModuleMatcher {self.name} {self.modules!r}>'

    def info(self) -> list[str]:
        """A list of strings for displaying when dumping state."""
        return self.modules

    def match(self, module_name: str) -> bool:
        """Does `module_name` indicate a module in one of our packages?"""
        if not module_name:
            return False
        for m in self.modules:
            if module_name.startswith(m):
                if module_name == m:
                    return True
                if module_name[len(m)] == '.':
                    return True
        return False