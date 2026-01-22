from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Type, cast
from .lazy_import import lazy_import
import textwrap
from breezy import (
from breezy.i18n import gettext
from . import errors, hooks, registry
from . import revision as _mod_revision
from . import trace
from . import transport as _mod_transport
def _get_full_stack_on(self):
    """Get a fully-qualified URL for the stack_on location."""
    if self._stack_on is None:
        return None
    if self._stack_on_pwd is None:
        return self._stack_on
    else:
        return urlutils.join(self._stack_on_pwd, self._stack_on)