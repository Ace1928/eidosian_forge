import contextlib
import sys
from typing import TYPE_CHECKING, Set, cast
from ..lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
from breezy.i18n import gettext
from .. import config, controldir, errors, lockdir
from .. import transport as _mod_transport
from ..trace import mutter, note, warning
from ..transport import do_catching_redirections, local
def __get_workingtree_format(self):
    if self._workingtree_format is None:
        from .workingtree import format_registry as wt_format_registry
        self._workingtree_format = wt_format_registry.get_default()
    return self._workingtree_format