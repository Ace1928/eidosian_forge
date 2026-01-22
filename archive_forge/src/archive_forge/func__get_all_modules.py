from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Type, cast
from .lazy_import import lazy_import
import textwrap
from breezy import (
from breezy.i18n import gettext
from . import errors, hooks, registry
from . import revision as _mod_revision
from . import trace
from . import transport as _mod_transport
def _get_all_modules(self):
    """Return a set of the modules providing objects."""
    modules = set()
    for name in self.keys():
        modules.add(self._get_module(name))
    for getter in self._extra_formats:
        modules.add(getter.get_module())
    return modules