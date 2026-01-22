import os
import sys
from io import BytesIO
from typing import Callable, Dict, Iterable, Tuple, cast
import configobj
import breezy
from .lazy_import import lazy_import
import errno
import fnmatch
import re
from breezy import (
from breezy.i18n import gettext
from . import (bedding, commands, errors, hooks, lazy_regex, registry, trace,
from .option import Option as CommandOption
class BranchOnlyStack(Stack):
    """Branch-only options stack."""

    def __init__(self, branch):
        bstore = branch._get_config_store()
        super().__init__([NameMatcher(bstore, None).get_sections], bstore)
        self.branch = branch

    def lock_write(self, token=None):
        return self.branch.lock_write(token)

    def unlock(self):
        return self.branch.unlock()

    def set(self, name, value):
        with self.lock_write():
            super().set(name, value)
            self.store.save_changes()

    def remove(self, name):
        with self.lock_write():
            super().remove(name)
            self.store.save_changes()