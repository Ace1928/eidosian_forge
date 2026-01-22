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
class BranchStack(Stack):
    """Per-location options falling back to branch then global options stack.

    The following sections are queried:

    * command-line overrides,

    * the sections matching ``location`` in ``locations.conf``, the order being
      defined by the number of path components in the section glob, higher
      numbers first (from most specific section to most generic),

    * the no-name section in branch.conf,

    * the ``DEFAULT`` section in ``bazaar.conf``.

    This stack will use the no-name section in ``branch.conf`` as its
    MutableSection.
    """

    def __init__(self, branch):
        lstore = self.get_shared_store(LocationStore())
        bstore = branch._get_config_store()
        gstore = self.get_shared_store(GlobalStore())
        super().__init__([self._get_overrides, LocationMatcher(lstore, branch.base).get_sections, NameMatcher(bstore, None).get_sections, NameMatcher(gstore, 'DEFAULT').get_sections], bstore)
        self.branch = branch

    def lock_write(self, token=None):
        return self.branch.lock_write(token)

    def unlock(self):
        return self.branch.unlock()

    def set(self, name, value):
        with self.lock_write():
            super().set(name, value)

    def remove(self, name):
        with self.lock_write():
            super().remove(name)