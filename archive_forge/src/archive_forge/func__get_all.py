from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Type, cast
from .lazy_import import lazy_import
import textwrap
from breezy import (
from breezy.i18n import gettext
from . import errors, hooks, registry
from . import revision as _mod_revision
from . import trace
from . import transport as _mod_transport
def _get_all(self):
    """Return all formats, even those not usable in metadirs."""
    result = []
    for getter in self._get_all_lazy():
        fmt = getter()
        if callable(fmt):
            fmt = fmt()
        result.append(fmt)
    return result